"""
synthetic_mo_oracle.py — ZDT1 and DTLZ2 discrete-pool oracles, interface-
compatible with DiscreteMOExcipientOracle/DiscreteADACoatingsOracle
(bounds(), objective_directions(), objective_names(), query_mo(),
__len__(), _X_raw/_Y_raw/_scaler/_queried), for Step 3 of the compose_batch
plan: cheap, fast-iterating synthetic domains with known Pareto-front
structure, used to shake out compose_batch's new failure modes (duplicate-
index gaming — already foreclosed structurally by compose_sandbox.py;
timeout; overfitting to a specific fitness signal, not applicable here
since this is direct campaign replay, not evolution) BEFORE spending real-
domain compute on coatings/mAb.

Both are standard synthetic MO benchmarks with analytically known Pareto
fronts, giving a gradient signal the real domains structurally lack (real
oracles' true optimum is unknown by construction) — useful specifically
for confirming the new compose_batch/DA-COREG infrastructure works AT ALL
before trusting any real-domain result built on it.

ZDT1 (2 objectives, both minimise): f1(x) = x1;
g(x) = 1 + 9*sum(x2..xd)/(d-1); f2(x) = g(x)*(1 - sqrt(f1/g)).
Pareto-optimal front: x2..xd = 0, f1 in [0,1], f2 = 1 - sqrt(f1).

DTLZ2 (3 objectives, all minimise), d = M + k - 1 with M=3, k=4 (d=6):
g(x_M) = sum((xi - 0.5)^2) for xi in the last k variables;
f1 = (1+g)*cos(x1*pi/2)*cos(x2*pi/2); f2 = (1+g)*cos(x1*pi/2)*sin(x2*pi/2);
f3 = (1+g)*sin(x1*pi/2). Pareto-optimal front: last k vars = 0.5 (g=0),
f1^2+f2^2+f3^2 = 1 (unit sphere octant).

Both use d small enough (6) to keep per-batch cost comparable to the real
domains' 16D/4D inputs, not scaled up to the traditional literature's
30D/12D defaults — this is a plumbing/mechanism test, not a benchmark
paper, so dimensionality is chosen for iteration speed.

ZDT3 (2 objectives, both minimise), added for v6: f1(x) = x1;
g(x) = 1 + 9*sum(x2..xd)/(d-1); f2(x) = g(x)*(1 - sqrt(f1/g) -
(f1/g)*sin(10*pi*f1)). The sin term gives ZDT3 a DISCONNECTED Pareto
front (5 separate convex segments) — deliberately a different front
topology from ZDT1's single connected curve and DTLZ2's connected
surface, so v6's training set spans genuinely different front
geometries, not just different noise levels on similar shapes.

v6 noise model: DiscreteSyntheticMOOracle instances built with
noise_cv > 0.0 (default 0.0, fully backward-compatible with every
existing caller of build_zdt1/build_dtlz2, none of which pass this arg)
apply i.i.d. multiplicative Gaussian noise to EACH objective on EVERY
query_mo call: y_observed = y_true * (1 + N(0, noise_cv)). Fresh noise
per call (not a fixed per-point value) — matches how the real
oracles (DiscreteMOExcipientOracle/DiscreteADACoatingsOracle) behave,
where re-observing the same physical point is not assumed to return an
identical reading. Ground truth (_Y_raw) is never mutated; noise is
applied only to the value returned to the caller, so the pool's true
Pareto structure stays fixed and known even though what evolved AFs
observe is noisy. Calibrated so an oracle's noise_cv roughly matches the
real domains' measured margin CVs (coatings ~15%, mAb ~50%) — see
v6/README.md for the exact per-training-domain assignment.
"""

import pathlib
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_SEED = 42
POOL_SIZE = 500


def _zdt1(X: np.ndarray) -> np.ndarray:
    f1 = X[:, 0]
    g = 1.0 + 9.0 * X[:, 1:].mean(axis=1)
    f2 = g * (1.0 - np.sqrt(np.clip(f1 / g, 0.0, None)))
    return np.column_stack([f1, f2])


def _zdt3(X: np.ndarray) -> np.ndarray:
    f1 = X[:, 0]
    g = 1.0 + 9.0 * X[:, 1:].mean(axis=1)
    ratio = np.clip(f1 / g, 0.0, None)
    f2 = g * (1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * f1))
    return np.column_stack([f1, f2])


def _dtlz2(X: np.ndarray, n_obj: int = 3) -> np.ndarray:
    d = X.shape[1]
    k = d - (n_obj - 1)
    x_head = X[:, :n_obj - 1]
    x_tail = X[:, n_obj - 1:]
    g = np.sum((x_tail - 0.5) ** 2, axis=1)
    F = []
    for m in range(n_obj):
        val = 1.0 + g
        for j in range(n_obj - 1 - m):
            val = val * np.cos(x_head[:, j] * np.pi / 2)
        if m > 0:
            val = val * np.sin(x_head[:, n_obj - 1 - m] * np.pi / 2)
        F.append(val)
    return np.column_stack(F)


class DiscreteSyntheticMOOracle:
    """
    Interface-compatible with DiscreteMOExcipientOracle/
    DiscreteADACoatingsOracle. Objectives are stored in their NATIVE
    minimise convention (both ZDT1 and DTLZ2 are minimisation problems in
    the standard literature formulation) — objective_directions() reports
    "min" for every objective, same as the coatings oracle's
    conductance_std column, so downstream to_allmax/pareto_front_of code
    (already parameterised on objective_directions()) handles the flip
    without any special-casing here.
    """

    def __init__(self, X_raw: np.ndarray, Y_raw: np.ndarray, objective_names: List[str],
                 noise_cv: float = 0.0, noise_seed: int = DATA_SEED):
        self._X_raw = X_raw
        self._Y_raw = Y_raw
        self._scaler = StandardScaler().fit(X_raw)
        self._queried = set()
        self._objective_names = objective_names
        # noise_cv=0.0 (default) is the ORIGINAL, fully-deterministic
        # behaviour every existing caller (v1-v5) relies on — no caller
        # before v6 passes this arg. See module docstring for the noise
        # model (multiplicative Gaussian, fresh draw per query_mo call,
        # applied only to the returned value, never to _Y_raw).
        self._noise_cv = noise_cv
        self._noise_rng = np.random.default_rng(noise_seed)

    @classmethod
    def build_zdt1(cls, d: int = 6, pool_size: int = POOL_SIZE,
                    seed: int = DATA_SEED, noise_cv: float = 0.0,
                    noise_seed: int = DATA_SEED) -> "DiscreteSyntheticMOOracle":
        rng = np.random.default_rng(seed)
        X_raw = rng.random((pool_size, d))
        Y_raw = _zdt1(X_raw)
        return cls(X_raw, Y_raw, objective_names=["f1", "f2"],
                    noise_cv=noise_cv, noise_seed=noise_seed)

    @classmethod
    def build_zdt3(cls, d: int = 6, pool_size: int = POOL_SIZE,
                    seed: int = DATA_SEED, noise_cv: float = 0.0,
                    noise_seed: int = DATA_SEED) -> "DiscreteSyntheticMOOracle":
        rng = np.random.default_rng(seed)
        X_raw = rng.random((pool_size, d))
        Y_raw = _zdt3(X_raw)
        return cls(X_raw, Y_raw, objective_names=["f1", "f2"],
                    noise_cv=noise_cv, noise_seed=noise_seed)

    @classmethod
    def build_dtlz2(cls, n_obj: int = 3, k: int = 4, pool_size: int = POOL_SIZE,
                     seed: int = DATA_SEED, noise_cv: float = 0.0,
                     noise_seed: int = DATA_SEED) -> "DiscreteSyntheticMOOracle":
        d = n_obj - 1 + k
        rng = np.random.default_rng(seed)
        X_raw = rng.random((pool_size, d))
        Y_raw = _dtlz2(X_raw, n_obj=n_obj)
        return cls(X_raw, Y_raw, objective_names=[f"f{i+1}" for i in range(n_obj)],
                    noise_cv=noise_cv, noise_seed=noise_seed)

    def __len__(self) -> int:
        return len(self._X_raw)

    def bounds(self) -> np.ndarray:
        return np.column_stack([self._X_raw.min(axis=0), self._X_raw.max(axis=0)])

    def objective_directions(self) -> List[str]:
        return ["min"] * len(self._objective_names)

    def objective_names(self) -> List[str]:
        return list(self._objective_names)

    def query_mo(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        x_s = self._scaler.transform(x.reshape(1, -1))[0]
        X_s = self._scaler.transform(self._X_raw)
        unqueried = [i for i in range(len(self._X_raw)) if i not in self._queried]
        if not unqueried:
            unqueried = list(range(len(self._X_raw)))
        dists = np.linalg.norm(X_s[unqueried] - x_s, axis=1)
        chosen = unqueried[int(np.argmin(dists))]
        self._queried.add(chosen)
        y_true = self._Y_raw[chosen].copy()
        if self._noise_cv <= 0.0:
            return y_true, chosen
        # Fresh multiplicative noise per call (see module docstring) —
        # _Y_raw itself is never touched, so repeated construction/replay
        # against the same seed still has a fixed, known ground truth.
        noise = self._noise_rng.normal(loc=0.0, scale=self._noise_cv, size=y_true.shape)
        return y_true * (1.0 + noise), chosen


if __name__ == "__main__":
    for name, oracle in [("ZDT1", DiscreteSyntheticMOOracle.build_zdt1()),
                          ("ZDT3", DiscreteSyntheticMOOracle.build_zdt3()),
                          ("DTLZ2", DiscreteSyntheticMOOracle.build_dtlz2())]:
        print(f"{name}: {len(oracle)} pool points, {oracle.objective_names()} "
              f"({oracle.objective_directions()})")
        print(f"  Y range:\n{np.column_stack([oracle._Y_raw.min(axis=0), oracle._Y_raw.max(axis=0)])}")

    # Noise calibration check: repeatedly query_mo the SAME pool point
    # (bypassing the without-replacement bookkeeping by resetting
    # _queried each call) and confirm the empirical CV of the OBSERVED
    # values matches the requested noise_cv — an end-to-end check of the
    # actual query_mo noise pathway, not just numpy's RNG in isolation.
    print("\nNoise calibration (repeated query_mo calls at one fixed point):")
    for name, builder, target_cv in [
        ("ZDT1@15%", DiscreteSyntheticMOOracle.build_zdt1, 0.15),
        ("DTLZ2-3@30%", DiscreteSyntheticMOOracle.build_dtlz2, 0.30),
        ("ZDT3@50%", DiscreteSyntheticMOOracle.build_zdt3, 0.50),
    ]:
        oracle = builder(noise_cv=target_cv)
        x0 = oracle._X_raw[0]
        y_true = oracle._Y_raw[0]
        observed = []
        for _ in range(3000):
            oracle._queried.clear()
            y_obs, idx = oracle.query_mo(x0)
            assert idx == 0
            observed.append(y_obs)
        observed = np.array(observed)
        empirical_cv = observed.std(axis=0) / np.abs(y_true)
        print(f"  {name}: target_cv={target_cv}  y_true={y_true.round(4)}  "
              f"empirical_cv={empirical_cv.round(4)}")
