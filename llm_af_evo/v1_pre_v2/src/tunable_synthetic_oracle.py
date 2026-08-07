"""
tunable_synthetic_oracle.py — a ZDT1-family, 2-objective, discrete-pool
oracle with THREE independently tunable knobs, designed to let front-range
normalisation (see docs/llm_evolved_afs_comprehensive_log.md, gen6_child0)
be confirmed or falsified on a domain we control, instead of diagnosed-but-
inconclusive on coatings/mAb.

Why this domain and not vanilla synthetic_mo_oracle.py's ZDT1/DTLZ2
--------------------------------------------------------------------
The thesis's own diagnosis of the two real domains gives two necessary
conditions for the confirmation experiment to be tractable, and a third
implicit one that front-range normalisation needs to have anything to bite
on at all:

1. NOT mu_sum-dominated (coatings' failure mode). If posterior-mean
   differences across the candidate pool already separate the ranking, the
   sigma term — normalised or not — never gets to influence which point is
   queried. `plateau_sharpness` below controls exactly this: it flattens
   the mean-response surface near the region that matters, forcing
   candidates to tie (or nearly tie) on mu_sum so sigma decides.
2. Noise floor low enough that GP posterior std reflects epistemic
   uncertainty, not measurement noise (mAb's failure mode, CV~49.7%).
   `noise_level` below is an explicit CV-like knob, defaulting far under
   that (~5-15%), with three heteroscedasticity shapes so you can also
   test whether the *shape* of the noise (not just its level) matters.
3. Objectives on genuinely different natural scales. This is the condition
   front-range normalisation is FOR — if f1 and f2 already live on
   comparable ranges, a raw-sigma UCB can't be distinguished from a
   front-range-normalised one. `scale2` sets objective 2's scale relative
   to objective 1 (default 20x), and `pareto_front_range` (computed from
   the live non-dominated set, same convention as the real oracles) is
   what the normalised AF divides by.

Base function is ZDT1 (2 objectives, both minimise):
    f1(x) = scale1 * x1
    g(x)  = 1 + 9 * mean(x2..xd)^plateau_sharpness
    f2(x) = scale2 * g(x) * (1 - sqrt(f1/(scale1*g(x))))
Pareto-optimal front: x2..xd = 0 (so g=1), f1 in [0, scale1],
f2 = scale2 * (1 - sqrt(f1/scale1)).

plateau_sharpness > 1 flattens g(x) near x2..xd = 0 — i.e. near the
Pareto-optimal region a WIDE band of x gives nearly identical g, hence
nearly identical mu_sum, which is exactly the "mu_sum doesn't already
separate the pool" condition coatings failed to provide. Sweep it; don't
guess it — see sweep_tunable_domain.py.

Interface-compatible with DiscreteMOExcipientOracle / DiscreteADACoatings-
Oracle / DiscreteSyntheticMOOracle (bounds(), objective_directions(),
objective_names(), query_mo(), __len__(), _X_raw/_Y_raw/_scaler/_queried),
so it drops into the existing replay/evolution harness unchanged. The one
addition is noise: query_mo() returns a NOISY draw, while _Y_raw / a new
true_y() accessor keep the noiseless ground truth for diagnostics — real
oracles can't offer that, which is exactly why this domain exists.
"""

import pathlib
from typing import List, Literal, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_SEED = 42
POOL_SIZE = 500

NoiseMode = Literal["homoscedastic", "proportional", "input_dependent"]


def _zdt1_tunable(X: np.ndarray, scale1: float, scale2: float,
                   plateau_sharpness: float) -> np.ndarray:
    """Noiseless ground truth. X assumed in [0,1]^d."""
    f1 = scale1 * X[:, 0]
    g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1) ** plateau_sharpness
    f2 = scale2 * g * (1.0 - np.sqrt(np.clip(f1 / (scale1 * g), 0.0, None)))
    return np.column_stack([f1, f2])


def _dist_to_pareto_front(f1: np.ndarray, scale1: float, scale2: float) -> np.ndarray:
    """Distance (in f2 units) from each point's f1 to the analytic Pareto
    curve f2 = scale2*(1 - sqrt(f1/scale1)), used only by the
    'input_dependent' noise shape to make noise correlate with how close a
    candidate is to the optimum (a plausible real-assay pattern: cleaner
    signal far from the operating envelope, noisier near it, or vice
    versa — sign is controlled by boundary_gain)."""
    f1_clipped = np.clip(f1, 0.0, scale1)
    return scale2 * (1.0 - np.sqrt(f1_clipped / scale1))


class TunableSyntheticMOOracle:
    """
    Parameters
    ----------
    scale1, scale2 : float
        Native scale of objective 1 / objective 2. Keep these different
        (default 1.0 / 3.0) — this is the condition front-range
        normalisation needs to have any effect to demonstrate. NOTE: raw
        mu_sum is NOT scale-invariant the way sigma_norm is, so pushing
        scale2 much above ~3-5x makes mu_sum dominate the AF score by
        sheer magnitude regardless of plateau_sharpness — see
        sweep_tunable_domain.py's dominance_ratio, which showed scale2=20
        driving the ratio into the hundreds even at plateau_sharpness=3.
        Re-run the sweep before changing this default.
    plateau_sharpness : float
        Exponent on g(x)'s driving variable. 1.0 = vanilla ZDT1. >1
        flattens the mean-response surface near the Pareto-optimal region,
        suppressing mu_sum's ability to separate candidates there — the
        knob that avoids coatings' mu_sum-dominance failure mode. Sweep
        upward from 1.0 (try 1.5, 2.0, 3.0) and check the diagnostic in
        sweep_tunable_domain.py rather than assuming a value works.
    noise_level : float
        CV-like fraction (0.05 = 5%, 0.15 = 15%). Keep well under mAb's
        ~0.497 — that's the whole point. Applies per noise_mode below.
    noise_mode : {"homoscedastic", "proportional", "input_dependent"}
        homoscedastic:   sigma_obs_i = noise_level * scale_i (constant)
        proportional:    sigma_obs_i(x) = noise_level * |f_i(x)|  (constant
                          CV, classic heteroscedastic assay noise)
        input_dependent: sigma_obs_i(x) = noise_level * scale_i *
                          (1 + boundary_gain * dist_to_front(x)/scale_i)
                          — noise itself varies with location, not just
                          magnitude of f. Use this to test whether AFs
                          (front-range-normalised or not) cope with noise
                          that isn't simply proportional to signal.
    boundary_gain : float
        Only used by "input_dependent". Positive = noisier far from the
        Pareto front (cleaner near the optimum, a common real pattern:
        well-characterised operating region vs. unexplored fringe).
        Negative = noisier near the front instead.
    """

    def __init__(self, X_raw: np.ndarray, Y_true: np.ndarray,
                 objective_names: List[str], scale1: float, scale2: float,
                 noise_level: float, noise_mode: NoiseMode,
                 boundary_gain: float, seed: int):
        self._X_raw = X_raw
        self._Y_true = Y_true          # noiseless ground truth, kept for diagnostics
        self._scaler = StandardScaler().fit(X_raw)
        self._queried = set()
        self._objective_names = objective_names
        self._scale1 = scale1
        self._scale2 = scale2
        self._noise_level = noise_level
        self._noise_mode = noise_mode
        self._boundary_gain = boundary_gain
        self._rng = np.random.default_rng(seed + 1)  # separate stream from X sampling
        # _Y_raw: ONE noisy realisation per pool point, drawn once here —
        # not resampled on every query_mo() call. Matches the semantics of
        # the real oracles this is interface-compatible with (DiscreteADA-
        # CoatingsOracle/DiscreteMOExcipientOracle's _Y_raw is fixed
        # experimental data, not something that changes if you "look at it
        # again"), and is REQUIRED for make_shared_inits() in
        # excipient_campaign_mo.py, which indexes disc_oracle._Y_raw[idx]
        # directly rather than going through query_mo() for a campaign's
        # initial points. Before this fix, initial points bypassed the
        # noise model entirely (AttributeError: no _Y_raw) and any later
        # query_mo() call for the same point would silently return a
        # DIFFERENT noisy value than a previous call — not a physical
        # measurement's behaviour.
        obs_stds = np.array([self._obs_std(i) for i in range(len(self._Y_true))])
        self._Y_raw = self._Y_true + self._rng.normal(0.0, obs_stds)

    @classmethod
    def build(cls, d: int = 6, pool_size: int = POOL_SIZE,
              scale1: float = 1.0, scale2: float = 3.0,
              plateau_sharpness: float = 3.0,
              noise_level: float = 0.08,
              noise_mode: NoiseMode = "proportional",
              boundary_gain: float = 0.5,
              seed: int = DATA_SEED) -> "TunableSyntheticMOOracle":
        rng = np.random.default_rng(seed)
        X_raw = rng.random((pool_size, d))
        Y_true = _zdt1_tunable(X_raw, scale1, scale2, plateau_sharpness)
        return cls(X_raw, Y_true, objective_names=["f1", "f2"],
                    scale1=scale1, scale2=scale2, noise_level=noise_level,
                    noise_mode=noise_mode, boundary_gain=boundary_gain, seed=seed)

    def __len__(self) -> int:
        return len(self._X_raw)

    def bounds(self) -> np.ndarray:
        return np.column_stack([self._X_raw.min(axis=0), self._X_raw.max(axis=0)])

    def objective_directions(self) -> List[str]:
        return ["min"] * len(self._objective_names)

    def objective_names(self) -> List[str]:
        return list(self._objective_names)

    def true_y(self, idx: int) -> np.ndarray:
        """Noiseless ground truth for candidate idx — diagnostics only,
        never fed to the AF/GP. Real oracles can't offer this; use it to
        sanity-check that a chosen (noise_level, plateau_sharpness) pair
        actually produces a domain in the target regime before spending
        LLM-evolution compute on it."""
        return self._Y_true[idx].copy()

    def _obs_std(self, idx: int) -> np.ndarray:
        f_true = self._Y_true[idx]
        scales = np.array([self._scale1, self._scale2])
        if self._noise_mode == "homoscedastic":
            return self._noise_level * scales
        if self._noise_mode == "proportional":
            return self._noise_level * np.abs(f_true)
        if self._noise_mode == "input_dependent":
            dist = _dist_to_pareto_front(np.array([f_true[0]]), self._scale1, self._scale2)[0]
            return self._noise_level * scales * (1.0 + self._boundary_gain * dist / self._scale2)
        raise ValueError(f"unknown noise_mode {self._noise_mode!r}")

    def query_mo(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        x_s = self._scaler.transform(x.reshape(1, -1))[0]
        X_s = self._scaler.transform(self._X_raw)
        unqueried = [i for i in range(len(self._X_raw)) if i not in self._queried]
        if not unqueried:
            unqueried = list(range(len(self._X_raw)))
        dists = np.linalg.norm(X_s[unqueried] - x_s, axis=1)
        chosen = unqueried[int(np.argmin(dists))]
        self._queried.add(chosen)
        return self._Y_raw[chosen].copy(), chosen


if __name__ == "__main__":
    for plateau in (1.0, 2.0, 3.0):
        oracle = TunableSyntheticMOOracle.build(plateau_sharpness=plateau)
        print(f"plateau_sharpness={plateau}: {len(oracle)} pool points, "
              f"{oracle.objective_names()} ({oracle.objective_directions()})")
        print(f"  true Y range:\n{np.column_stack([oracle._Y_true.min(axis=0), oracle._Y_true.max(axis=0)])}")
