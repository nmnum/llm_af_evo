"""
synthetic_oracles.py — Discrete synthetic benchmark oracles for the router study.

Creates NNOracle instances from LHS-sampled synthetic functions.
Hartmann3 and Hartmann6 provide structured multi-modal landscapes
that complement the ADA datasets.

Usage:
    from synthetic_oracles import make_hartmann3_oracle, make_hartmann6_oracle

    oracle = make_hartmann3_oracle(n_points=100, seed=42)
    # Returns a standard NNOracle with .query(), .global_best(), .bounds()
    # Drop-in replacement for NNOracle.from_dataset()

To save as CSV (for use with existing pipeline):
    python synthetic_oracles.py --out_dir data/
"""

import argparse
import pathlib
import numpy as np
import pandas as pd


# ── Objective functions ───────────────────────────────────────────────────────

def hartmann3(x: np.ndarray) -> float:
    """Hartmann 3D: 4 local optima, global min ≈ -3.86 at (0.114, 0.556, 0.852).
    We MAXIMISE, so return negative.
    x in [0,1]^3
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [3.0, 10.0, 30.0],
        [0.1,  10.0, 35.0],
        [3.0,  10.0, 30.0],
        [0.1,  10.0, 35.0],
    ])
    P = 1e-4 * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381,  5743, 8828],
    ])
    result = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i]) ** 2)
        result += alpha[i] * np.exp(-inner)
    return float(result)  # maximise (positive = better)


def hartmann6(x: np.ndarray) -> float:
    """Hartmann 6D: 6 local optima, global min ≈ -3.32 at known location.
    We MAXIMISE, so return negative of the standard minimisation form.
    x in [0,1]^6
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10.0, 3.0,  17.0, 3.5,  1.7,  8.0],
        [0.05, 10.0, 17.0, 0.1,  8.0,  14.0],
        [3.0,  3.5,  1.7,  10.0, 17.0, 8.0],
        [17.0, 8.0,  0.05, 10.0, 0.1,  14.0],
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124,  8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ])
    result = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i]) ** 2)
        result += alpha[i] * np.exp(-inner)
    return float(result)  # maximise


# ── Oracle factory ────────────────────────────────────────────────────────────

def _make_discrete_oracle(obj_fn, d: int, n_points: int, seed: int):
    """
    Sample n_points from [0,1]^d using LHS, evaluate obj_fn, return NNOracle.
    The oracle is in raw (unscaled) space — bounds are [0,1] per dimension.
    """
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from oracle import NNOracle

    rng = np.random.default_rng(seed)

    # Latin Hypercube Sampling for good coverage
    cuts = np.linspace(0, 1, n_points + 1)
    X = np.zeros((n_points, d))
    for j in range(d):
        perm = rng.permutation(n_points)
        u = rng.uniform(size=n_points)
        X[:, j] = cuts[perm] + u / n_points

    y = np.array([obj_fn(X[i]) for i in range(n_points)])
    return NNOracle(X, y)


def make_hartmann3_oracle(n_points: int = 100, seed: int = 42):
    """NNOracle wrapping Hartmann3 (3D, 4 optima)."""
    return _make_discrete_oracle(hartmann3, d=3, n_points=n_points, seed=seed)


def make_hartmann6_oracle(n_points: int = 100, seed: int = 42):
    """NNOracle wrapping Hartmann6 (6D, 6 optima)."""
    return _make_discrete_oracle(hartmann6, d=6, n_points=n_points, seed=seed)


# ── CSV export (for NNOracle.from_dataset compatibility) ────────────────────

def _oracle_to_csv(oracle, name: str, out_dir: pathlib.Path):
    """
    Save oracle data as CSV in the format expected by NNOracle.from_dataset.
    Adds a 'synthetic_<name>' dataset type that can be loaded via a custom
    factory (see oracle.py update below).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    d = oracle._X_raw.shape[1]
    cols = {f"x{i}": oracle._X_raw[:, i] for i in range(d)}
    cols["y"] = oracle._y_raw
    df = pd.DataFrame(cols)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df)} rows, {d}D, best={oracle.global_best():.4f})")
    return path


# ── oracle.py extension (add to NNOracle.from_dataset) ───────────────────────

ORACLE_PATCH = '''
    @classmethod
    def _from_synthetic_csv(cls, csv_path: pathlib.Path) -> "NNOracle":
        """Load a synthetic benchmark oracle from CSV (x0, x1, ..., y columns)."""
        df = pd.read_csv(csv_path)
        y_col = "y"
        x_cols = [c for c in df.columns if c != y_col]
        X = df[x_cols].values.astype(float)
        y = df[y_col].values.astype(float)
        return cls(X, y)
'''

ORACLE_FROM_DATASET_ADDITION = '''
        if dataset_name.startswith("synthetic_"):
            stem = dataset_name[len("synthetic_"):]
            csv_path = data_dir / f"{stem}.csv"
            return cls._from_synthetic_csv(csv_path)
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data",
                        help="Directory to save synthetic CSVs")
    parser.add_argument("--n_points", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)

    print("Generating synthetic oracles...")
    h3 = make_hartmann3_oracle(n_points=args.n_points, seed=args.seed)
    h6 = make_hartmann6_oracle(n_points=args.n_points, seed=args.seed)

    _oracle_to_csv(h3, "hartmann3", out_dir)
    _oracle_to_csv(h6, "hartmann6", out_dir)

    print()
    print("Hartmann3: 3D, 4 optima")
    print(f"  global_best = {h3.global_best():.4f}")
    print(f"  bounds      = {h3.bounds()}")
    print()
    print("Hartmann6: 6D, 6 optima")
    print(f"  global_best = {h6.global_best():.4f}")
    print(f"  bounds      = {h6.bounds()}")
    print()
    print("To load in NNOracle.from_dataset, add to oracle.py:")
    print(ORACLE_FROM_DATASET_ADDITION)
