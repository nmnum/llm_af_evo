"""
controllers/mock_controller.py — Rule-based mock LLM controllers.

These replicate the decision logic of the real LLM controllers without
requiring Ollama. Used for pipeline validation and reproducible benchmarking.

MockApproachAController : heuristic β-tuning based on improvement rate
MockApproachBController : phase-based strategy switching
MockApproachCController : inline code variants (no subprocess overhead)
"""

import numpy as np
from typing import Any, Dict


class MockApproachAController:
    """
    Mock Approach A: heuristic UCB β-tuning.

    Rule: if improvement_rate > 0.3, exploit (low β); else explore (high β).
    Adds small random noise to simulate LLM variability.
    """

    _needs_gp_uncertainty = False

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._beta = 1.0

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        rate = context.get("improvement_rate", 0.5)
        progress = context["step"] / context["budget"]

        if rate > 0.3:
            # Good progress — exploit
            target_beta = 0.2 + self._rng.uniform(-0.05, 0.05)
        elif progress < 0.3:
            # Early phase — explore
            target_beta = 400.0 + self._rng.uniform(-50, 50)
        else:
            # Mid/late phase — balanced
            target_beta = 20.0 + self._rng.uniform(-5, 5)

        self._beta = float(np.clip(target_beta, 0.01, 1000.0))
        return {"strategy": "ucb", "params": {"beta": self._beta}}


class MockApproachBController:
    """
    Mock Approach B: phase-based strategy switching.

    Phase logic:
      0–30%  : LHS (space-filling exploration)
      30–60% : EI (model-based exploitation)
      60–80% : UCB β=20 (balanced)
      80–100%: UCB β=0.2 (pure exploitation)

    Switches strategy when phase changes.
    """

    _needs_gp_uncertainty = False

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._current_strategy = "lhs"
        self._current_params: Dict = {}

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        progress = context["step"] / context["budget"]

        if progress < 0.30:
            strategy, params = "lhs", {}
        elif progress < 0.60:
            strategy, params = "ei", {}
        elif progress < 0.80:
            strategy, params = "ucb", {"beta": 20.0}
        else:
            strategy, params = "ucb", {"beta": 0.2}

        self._current_strategy = strategy
        self._current_params = params
        return {"strategy": strategy, "params": params}


class MockApproachCController:
    """
    Mock Approach C: inline code variants.

    Simulates code rewriting by switching between three inline implementations:
      Phase 1 (0–40%): random search
      Phase 2 (40–70%): UCB with β=5
      Phase 3 (70–100%): EI

    Returns x_next directly (no subprocess) for speed.
    """

    _needs_gp_uncertainty = False

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        progress = context["step"] / context["budget"]
        X_obs = context["X_obs"]
        y_obs = context["y_obs"]
        bounds = context["bounds"]

        try:
            if progress < 0.40:
                x_next = self._random_suggest(bounds)
            elif progress < 0.70:
                x_next = self._ucb_suggest(X_obs, y_obs, bounds, beta=5.0)
            else:
                x_next = self._ei_suggest(X_obs, y_obs, bounds)

            x_next = np.clip(x_next, bounds[:, 0], bounds[:, 1])
            return {"strategy": "_custom", "params": {}, "x_next": x_next}

        except Exception:
            return {"strategy": "random", "params": {}}

    def _random_suggest(self, bounds: np.ndarray) -> np.ndarray:
        d = bounds.shape[0]
        return np.array([
            self._rng.uniform(bounds[i, 0], bounds[i, 1]) for i in range(d)
        ])

    def _ucb_suggest(
        self, X_obs: np.ndarray, y_obs: np.ndarray, bounds: np.ndarray, beta: float = 1.0
    ) -> np.ndarray:
        import warnings
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        from sklearn.preprocessing import StandardScaler

        if len(X_obs) < 3:
            return self._random_suggest(bounds)

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_obs)
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
            alpha=1e-6, normalize_y=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X_s, y_obs)

        d = bounds.shape[0]
        candidates = np.column_stack([
            self._rng.uniform(bounds[i, 0], bounds[i, 1], 200) for i in range(d)
        ])
        mu, sigma = gp.predict(scaler.transform(candidates), return_std=True)
        return candidates[np.argmax(mu + beta * sigma)]

    def _ei_suggest(
        self, X_obs: np.ndarray, y_obs: np.ndarray, bounds: np.ndarray
    ) -> np.ndarray:
        import warnings
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import norm

        if len(X_obs) < 3:
            return self._random_suggest(bounds)

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_obs)
        y_mean, y_std = y_obs.mean(), y_obs.std() + 1e-12
        y_s = (y_obs - y_mean) / y_std

        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
            alpha=1e-6, normalize_y=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X_s, y_s)

        y_best_s = (y_obs.max() - y_mean) / y_std
        d = bounds.shape[0]
        candidates = np.column_stack([
            self._rng.uniform(bounds[i, 0], bounds[i, 1], 200) for i in range(d)
        ])
        mu, sigma = gp.predict(scaler.transform(candidates), return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        z = (mu - y_best_s) / sigma
        acq = sigma * (z * norm.cdf(z) + norm.pdf(z))
        return candidates[np.argmax(acq)]
