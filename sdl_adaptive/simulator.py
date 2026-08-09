"""
simulator.py — Campaign simulator for SDL strategy evaluation.

The simulator replays an experimental campaign step-by-step:
  1. Initialise with n_init random points from the oracle.
  2. At each subsequent step, ask the active strategy for the next point.
  3. Every controller_interval steps, call the controller to (optionally)
     switch strategy or adjust parameters.
  4. Record running best, decisions, and failures.

GP uncertainty is computed lazily — only when a controller sets
_needs_gp_uncertainty = True (real LLM controllers). Mock/fixed baselines
use a cheap proxy (normalised y std) to avoid the GP fitting overhead.
"""

import logging

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from oracle import NNOracle
from strategies import STRATEGY_MAP, DISCRETE_STRATEGY_MAP

logger = logging.getLogger(__name__)
_warned_unmapped_strategies: set = set()


def generate_init_points(oracle: NNOracle, n_init: int, seed: int, discrete: bool = False,
                          rng: Optional[np.random.Generator] = None):
    """Draw the n_init initialisation points CampaignSimulator.run() would draw
    for this (oracle, seed, discrete) combination, without running a campaign.

    Factored out of CampaignSimulator.run() so any caller that needs to run a
    *different* campaign loop against the identical shared init points that
    run_experiment.py's other conditions see at the same seed — e.g. handing
    off to shared_seed_experiment.py's run_egbo_campaign(), which owns its own
    batch loop rather than going through CampaignSimulator — can do so without
    duplicating (and risking drifting from) this logic.

    Pass `rng` when the caller already has a `np.random.default_rng(seed)` it
    needs to keep drawing from afterwards (this is exactly what
    CampaignSimulator.run() does — its post-init random fallbacks must
    continue the *same* generator's sequence, not restart it, or every
    existing seed's downstream results would silently change). Omit `rng`
    for a standalone call (e.g. from run_experiment.py before handing off to
    run_egbo_campaign, which owns its own separate rng thereafter) — a fresh
    `np.random.default_rng(seed)` is created and used once, matching what
    CampaignSimulator.run() draws at the very start of a campaign for the
    same seed.

    Returns
    -------
    (X_obs, y_obs, queried_indices) — queried_indices is None in continuous mode.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    bounds = oracle.bounds()
    d = bounds.shape[0]

    if discrete:
        N = len(oracle._X_raw)
        init_indices = rng.choice(N, size=n_init, replace=False).tolist()
        queried_indices = set(init_indices)
        X_obs = oracle._X_raw[init_indices].copy()
        y_obs = oracle._y_raw[init_indices].copy()
    else:
        queried_indices = None
        X_obs = np.array([
            [rng.uniform(bounds[i, 0], bounds[i, 1]) for i in range(d)]
            for _ in range(n_init)
        ])
        y_obs = np.array([oracle.query(x) for x in X_obs])

    return X_obs, y_obs, queried_indices


class CampaignSimulator:
    """
    Parameters
    ----------
    oracle            : NNOracle instance
    dataset_name      : string label (for logging)
    n_init            : number of random initialisation points
    budget            : total number of experiment steps
    controller_interval : call controller every k steps
    seed              : random seed for reproducibility
    """

    def __init__(
        self,
        oracle: NNOracle,
        dataset_name: str,
        n_init: int = 5,
        budget: int = 91,
        controller_interval: int = 5,
        seed: int = 0,
        discrete: bool = False,
    ):
        self.oracle = oracle
        self.dataset_name = dataset_name
        self.n_init = n_init
        self.budget = budget
        self.controller_interval = controller_interval
        self.seed = seed
        self.discrete = discrete

    def run(self, controller) -> Dict[str, Any]:
        """
        Run one full campaign.

        Returns
        -------
        dict with keys:
          running_best  : list[float], length = budget
          decisions     : list[(step, strategy_name, params)]
          failures      : list[int]  (steps where strategy raised an exception)
        """
        rng = np.random.default_rng(self.seed)
        # NOTE — thread-safety caveat, not fixed here: strategies.py's strategy
        # functions (ucb/ei/pi/thompson/random/lhs) call the numpy global RNG
        # directly rather than accepting `rng`, so this global seed call is a
        # race condition if multiple seeds/conditions run concurrently in threads
        # within one process. It is safe across separate OS processes (each gets
        # its own global numpy state), which is the lowest-effort way to
        # parallelise runs today — see run_experiment.py's --conditions flag for
        # backgrounding multiple single-condition runs. Threading this rng through
        # every strategy function is the real fix but touches all of strategies.py.
        np.random.seed(self.seed)

        bounds = self.oracle.bounds()
        d = bounds.shape[0]

        # ── Initialisation ──────────────────────────────────────────────
        # (module-level generate_init_points(), not inlined here, so other
        # campaign loops — e.g. run_egbo_campaign() invoked from
        # run_experiment.py — draw the identical shared init points at a
        # given seed. `rng` is passed through explicitly so it keeps
        # advancing from the same generator the post-init code below reads
        # from — creating a second independent generator here would replay
        # the same draws twice and silently change every existing seed's
        # downstream random fallbacks.)
        X_obs, y_obs, queried_indices = generate_init_points(
            self.oracle, self.n_init, self.seed, self.discrete, rng=rng
        )

        running_best = [float(y_obs.max())] * self.n_init
        decisions: List[Tuple] = []
        failures: List[int] = []

        # ── Active strategy state ────────────────────────────────────────
        current_strategy = "random"
        current_params: Dict = {}
        custom_fn = None  # set by approach C

        # ── Main loop ───────────────────────────────────────────────────
        for step in range(self.n_init, self.budget):

            # Controller decision every k steps
            if (step - self.n_init) % self.controller_interval == 0:
                context = self._build_context(
                    step, X_obs, y_obs, bounds, controller
                )
                try:
                    decision = controller.decide(context)
                except Exception as e:
                    decision = {"strategy": current_strategy, "params": current_params}

                strategy_name = decision.get("strategy", current_strategy)
                params = decision.get("params", current_params)
                x_next_custom = decision.get("x_next", None)  # approach C's first-step suggestion

                if strategy_name == "_custom":
                    # NOTE: x_next_custom below is only used for the first step of this
                    # controller_interval window. For approach C, controller.suggest_point()
                    # re-runs the LLM's accepted code against fresh X_obs/y_obs on every
                    # subsequent step in the window (see below) — previously the code ran
                    # once and the other controller_interval-1 steps silently fell back to
                    # random because "_custom" isn't in STRATEGY_MAP.
                    current_strategy = "_custom"
                    current_params = params
                elif strategy_name in STRATEGY_MAP:
                    current_strategy = strategy_name
                    current_params = params
                    custom_fn = None
                else:
                    # Controller chose a strategy name that isn't in
                    # STRATEGY_MAP/DISCRETE_STRATEGY_MAP at all (this is
                    # separate from "egbo"/"novelty_egbo" — those ARE mapped,
                    # but degrade to a raised RuntimeError, caught below and
                    # logged in `failures`, when EGBO's deps aren't importable
                    # in this interpreter — see strategies.py::_egbo_module).
                    # An unmapped name here is a silent no-op: the decision is
                    # discarded and the *previous* strategy keeps running,
                    # which can read in results as e.g. "random" even though
                    # nothing chose random.
                    self._warn_unmapped_strategy(strategy_name)

                decisions.append((step, current_strategy, {**current_params}))

            # ── Suggest next point ───────────────────────────────────────
            try:
                if self.discrete and current_strategy != "_custom":
                    # Score only unqueried rows — no snap needed
                    all_indices = np.arange(len(self.oracle._X_raw))
                    unqueried = [i for i in all_indices if i not in queried_indices]
                    if not unqueried:
                        # Exhausted dataset — fall back to random from full set
                        unqueried = list(all_indices)
                    X_pool = self.oracle._X_raw[unqueried]
                    if current_strategy not in DISCRETE_STRATEGY_MAP:
                        self._warn_unmapped_strategy(current_strategy)
                    strategy_fn = DISCRETE_STRATEGY_MAP.get(
                        current_strategy, DISCRETE_STRATEGY_MAP["random"]
                    )
                    pool_idx = strategy_fn(X_obs, y_obs, X_pool, **current_params)
                    chosen_global_idx = unqueried[pool_idx]
                    queried_indices.add(chosen_global_idx)
                    x_next = self.oracle._X_raw[chosen_global_idx]
                    y_next = float(self.oracle._y_raw[chosen_global_idx])

                elif current_strategy == "_custom":
                    if x_next_custom is not None:
                        # First step of this window: reuse the point already computed
                        # while parsing the LLM's decision (avoids re-running the sandbox).
                        x_next = np.asarray(x_next_custom, dtype=float)
                        x_next_custom = None  # consume once
                    elif hasattr(controller, "suggest_point"):
                        # Every subsequent step in the window: re-run the accepted code
                        # against the latest X_obs/y_obs so it actually governs all
                        # controller_interval steps, not just the first.
                        x_next = controller.suggest_point(X_obs, y_obs, bounds)
                        if x_next is None:
                            failures.append(step)
                            x_next = np.array([
                                rng.uniform(bounds[i, 0], bounds[i, 1]) for i in range(d)
                            ])
                    else:
                        x_next = np.array([
                            rng.uniform(bounds[i, 0], bounds[i, 1]) for i in range(d)
                        ])
                    x_next = np.clip(x_next, bounds[:, 0], bounds[:, 1])
                    if self.discrete:
                        # Snap to nearest *unqueried* row (not global nearest)
                        all_indices = np.arange(len(self.oracle._X_raw))
                        unqueried = [i for i in all_indices if i not in queried_indices]
                        if not unqueried:
                            unqueried = list(all_indices)
                        X_pool = self.oracle._X_raw[unqueried]
                        x_s = self.oracle._scaler.transform(x_next.reshape(1, -1))[0]
                        pool_scaled = self.oracle._scaler.transform(X_pool)
                        dists = np.linalg.norm(pool_scaled - x_s, axis=1)
                        pool_idx = int(np.argmin(dists))
                        chosen_global_idx = unqueried[pool_idx]
                        queried_indices.add(chosen_global_idx)
                        x_next = self.oracle._X_raw[chosen_global_idx]
                        y_next = float(self.oracle._y_raw[chosen_global_idx])
                    else:
                        y_next = self.oracle.query(x_next)

                else:
                    if current_strategy not in STRATEGY_MAP:
                        self._warn_unmapped_strategy(current_strategy)
                    strategy_fn = STRATEGY_MAP.get(current_strategy, STRATEGY_MAP["random"])
                    x_next = strategy_fn(X_obs, y_obs, bounds, **current_params)
                    y_next = self.oracle.query(x_next)

            except Exception as e:
                # Fallback to random on any strategy failure
                failures.append(step)
                if self.discrete and queried_indices is not None:
                    all_indices = np.arange(len(self.oracle._X_raw))
                    unqueried = [i for i in all_indices if i not in queried_indices]
                    if not unqueried:
                        unqueried = list(all_indices)
                    chosen_global_idx = int(rng.choice(unqueried))
                    queried_indices.add(chosen_global_idx)
                    x_next = self.oracle._X_raw[chosen_global_idx]
                    y_next = float(self.oracle._y_raw[chosen_global_idx])
                else:
                    x_next = np.array([
                        rng.uniform(bounds[i, 0], bounds[i, 1]) for i in range(d)
                    ])
                    y_next = self.oracle.query(x_next)

            X_obs = np.vstack([X_obs, x_next])
            y_obs = np.append(y_obs, y_next)
            running_best.append(float(y_obs.max()))

        return {
            "running_best": running_best,
            "decisions": decisions,
            "failures": failures,
            "X_obs": X_obs,
            "y_obs": y_obs,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        step: int,
        X_obs: np.ndarray,
        y_obs: np.ndarray,
        bounds: np.ndarray,
        controller,
    ) -> Dict[str, Any]:
        """Build the context dict passed to controller.decide()."""
        needs_gp = getattr(controller, "_needs_gp_uncertainty", False)

        if needs_gp:
            gp_uncertainty, lengthscale_norm = self._estimate_gp_uncertainty(
                X_obs, y_obs, bounds)
        else:
            # Cheap proxy: normalised std of observed y
            y_range = y_obs.max() - y_obs.min() + 1e-12
            gp_uncertainty = float(y_obs.std() / y_range)
            lengthscale_norm = 0.5  # unknown — use neutral value

        return {
            "step": step,
            "budget": self.budget,
            "n_obs": len(y_obs),
            "n_dims": bounds.shape[0],
            "best_so_far": float(y_obs.max()),
            "best_normalised": float(y_obs.max() / (self.oracle.global_best() + 1e-12)),
            "improvement_rate": self._improvement_rate(y_obs),
            "gp_uncertainty": gp_uncertainty,
            "lengthscale_norm": lengthscale_norm,
            "dataset": self.dataset_name,
            "X_obs": X_obs,
            "y_obs": y_obs,
            "bounds": bounds,
        }

    @staticmethod
    def _warn_unmapped_strategy(name: str) -> None:
        """A controller (e.g. approach_d) chose a strategy name that isn't in
        STRATEGY_MAP/DISCRETE_STRATEGY_MAP — e.g. 'egbo', which is only implemented
        in shared_seed_experiment.py's batch runner, not here. This silently falls
        back to random, which is a real confound if it happens a lot: previously
        there was no log signal to distinguish "controller chose random" from
        "controller chose something unimplemented that became random". Logged once
        per distinct name per process to avoid spamming a long campaign.
        """
        if name in _warned_unmapped_strategies:
            return
        _warned_unmapped_strategies.add(name)
        logger.warning(
            f"Strategy '{name}' is not implemented in this simulator's "
            f"STRATEGY_MAP/DISCRETE_STRATEGY_MAP — every step choosing it silently "
            f"falls back to random. If a controller can select '{name}', either "
            f"implement it here or remove it from that controller's valid strategies."
        )

    @staticmethod
    def _improvement_rate(y_obs: np.ndarray, window: int = 10) -> float:
        """Fraction of recent steps that improved the running best."""
        if len(y_obs) < 2:
            return 1.0
        recent = y_obs[-window:]
        improvements = sum(
            recent[i] > recent[:i].max() for i in range(1, len(recent))
        )
        return float(improvements / max(len(recent) - 1, 1))

    @staticmethod
    def _estimate_gp_uncertainty(
        X_obs: np.ndarray, y_obs: np.ndarray, bounds: np.ndarray,
        n_test: int = 100,
    ) -> tuple:
        """Mean GP posterior std and normalised lengthscale over random test points.

        Returns
        -------
        (gp_uncertainty, lengthscale_norm) : (float, float)
            gp_uncertainty  — mean posterior std over n_test random points
            lengthscale_norm — fitted Matern lengthscale / mean domain width,
                               in original input space (not scaled space).
                               High (>0.30) = smooth landscape, GP informative.
                               Low (<0.15)  = rough/flat landscape, GP unreliable.
        """
        import warnings
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_obs)
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
            alpha=1e-6,
            normalize_y=True,
            # Match strategies.py's fit_gp (n_restarts_optimizer=2) — this GP's
            # signals (gp_uncertainty, lengthscale_norm) are fed straight into the
            # LLM's prompt, so an unconverged fit (sklearn defaults to 0 restarts)
            # means the LLM is reasoning on a noisy kernel fit.
            n_restarts_optimizer=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X_s, y_obs)

        d = bounds.shape[0]
        test_pts = np.column_stack([
            np.random.uniform(bounds[i, 0], bounds[i, 1], n_test) for i in range(d)
        ])
        _, sigma = gp.predict(scaler.transform(test_pts), return_std=True)

        # Lengthscale in original space: ls_scaled * mean(scaler.scale_)
        ls_scaled = float(gp.kernel_.length_scale)
        mean_scale = float(scaler.scale_.mean())
        mean_domain_width = float((bounds[:, 1] - bounds[:, 0]).mean())
        ls_original = ls_scaled * mean_scale
        lengthscale_norm = ls_original / (mean_domain_width + 1e-12)
        lengthscale_norm = float(np.clip(lengthscale_norm, 0.0, 10.0))

        return float(sigma.mean()), lengthscale_norm
