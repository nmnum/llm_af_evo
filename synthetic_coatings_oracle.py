"""
synthetic_coatings_oracle.py — Synthetic coatings oracle for Phase 3 generalisability test.

Mimics the Ada pareto_20210112 dataset structure (4D continuous, single objective)
so we can test whether the LS-NA-EGBO architecture generalises to a different domain
with only prompt changes (no code changes).

The landscape is designed to have:
  - Smooth structure (GP-friendly)
  - A clear optimum in the mid-range
  - Interaction effects (high x0 + high x1 = bad)
  - Noise (matching real experimental conditions)

This is NOT a substitute for the real Ada data — it's a synthetic benchmark
for testing architectural generalisability.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class SyntheticCoatingsOracle:
    """
    4D continuous coatings optimization oracle.
    Inputs: x0 (concentration), x1 (temperature), x2 (flow_rate), x3 (pressure)
    All in [0, 1].
    Output: scalar coating quality score (higher = better).
    """

    def __init__(self, seed: int = 42, noise_level: float = 0.05):
        self.seed = seed
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)
        self._d = 4

    def _true_function(self, X):
        """
        Smooth landscape with interactions:
        - x0 (concentration): moderate values best, high values bad with high x1
        - x1 (temperature): mid-range optimal (0.4-0.6)
        - x2 (flow_rate): low-moderate best
        - x3 (pressure): low pressure better
        - Interaction: high x0 + high x1 → penalty (cracking)
        """
        x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        # Main effects (Gaussian peaks)
        f0 = np.exp(-0.5 * ((x0 - 0.4) / 0.2) ** 2)
        f1 = np.exp(-0.5 * ((x1 - 0.5) / 0.15) ** 2)
        f2 = np.exp(-0.5 * ((x2 - 0.3) / 0.2) ** 2)
        f3 = np.exp(-0.5 * ((x3 - 0.2) / 0.25) ** 2)

        # Combined quality (multiplicative — all need to be decent)
        quality = f0 * f1 * f2 * f3

        # Interaction penalty: high concentration + high temperature = cracking
        cracking_penalty = np.where(
            (x0 > 0.7) & (x1 > 0.8),
            0.3 * ((x0 - 0.7) * (x1 - 0.8) / 0.3),
            0.0
        )
        quality = quality * (1.0 - cracking_penalty)

        # Scale to [0, 1]
        return quality

    def make_discrete_oracle(self, n_samples=200, seed=42):
        """Create a discrete pool of samples (NNOracle-compatible)."""
        rng = np.random.default_rng(seed)
        # Latin hypercube sampling for good coverage
        X = np.zeros((n_samples, self._d))
        for j in range(self._d):
            perm = rng.permutation(n_samples)
            X[:, j] = (perm + rng.uniform(0, 1, n_samples)) / n_samples

        # Compute noise-free scores
        y_clean = self._true_function(X)
        # Add noise
        y_noisy = y_clean + rng.normal(0, self.noise_level, n_samples)
        y_noisy = np.clip(y_noisy, 0, 1)

        scaler = StandardScaler()
        scaler.fit(X)

        return DiscreteCoatingsOracle(X, y_noisy, y_clean, scaler)


class DiscreteCoatingsOracle:
    """NNOracle-compatible discrete oracle for coatings."""

    def __init__(self, X_raw, y_raw, y_clean, scaler):
        self._X_raw = X_raw
        self._y_raw = y_raw
        self._y_clean = y_clean
        self._scaler = scaler
        self._queried = set()
        self._d = X_raw.shape[1]

    def bounds(self):
        return np.column_stack([np.zeros(self._d), np.ones(self._d)])

    def query(self, x):
        """Query nearest neighbour. Returns (y_value, index)."""
        x_s = self._scaler.transform(x.reshape(1, -1))[0]
        X_all_s = self._scaler.transform(self._X_raw)

        # Find nearest unqueried point
        unqueried = [i for i in range(len(self._X_raw)) if i not in self._queried]
        if not unqueried:
            unqueried = list(range(len(self._X_raw)))

        dists = np.linalg.norm(X_all_s[unqueried] - x_s, axis=1)
        chosen = unqueried[int(np.argmin(dists))]
        self._queried.add(chosen)
        return float(self._y_raw[chosen]), chosen

    def global_best(self):
        return float(self._y_raw.max())

    def pareto_front_indices(self):
        """For single-objective, the 'Pareto front' is just the best point."""
        return np.array([int(np.argmax(self._y_raw))])
