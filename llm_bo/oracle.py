"""
oracle.py — Nearest-neighbour oracle for SDL campaign simulation.

Replaces a GP oracle that extrapolated to 193× the empirical maximum.
NNOracle always returns a real measured value from the ADA dataset,
so AUC is guaranteed to be bounded in [0, 1].
"""

import numpy as np
import pandas as pd
import pathlib
from sklearn.preprocessing import StandardScaler


class NNOracle:
    """
    Nearest-neighbour oracle: given a query point, returns the conductivity
    of the closest real ADA experimental point (in standardised input space).

    Never extrapolates — all returned values are real measurements.
    """

    def __init__(self, X_raw: np.ndarray, y_raw: np.ndarray):
        """
        Parameters
        ----------
        X_raw : (n, d) array of raw (unstandardised) input features
        y_raw : (n,) array of output values (conductivity or scalarised Pareto)
        """
        self._X_raw = X_raw.copy()
        self._y_raw = y_raw.copy()
        self._scaler = StandardScaler()
        self._X_scaled = self._scaler.fit_transform(X_raw)
        self._bounds = np.column_stack([X_raw.min(axis=0), X_raw.max(axis=0)])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, x: np.ndarray) -> float:
        """Return the output of the nearest training point to x."""
        x_s = self._scaler.transform(x.reshape(1, -1))[0]
        dists = np.linalg.norm(self._X_scaled - x_s, axis=1)
        return float(self._y_raw[np.argmin(dists)])

    def global_best(self) -> float:
        return float(self._y_raw.max())

    def bounds(self) -> np.ndarray:
        """Return (d, 2) array of [min, max] per dimension."""
        return self._bounds.copy()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset(cls, dataset_name: str, data_dir: str) -> "NNOracle":
        """
        Load an oracle from a named ADA dataset.

        Supported dataset_name values:
          - 'coatings'
          - 'pareto_<stem>'  where <stem> is the CSV filename without .csv
            e.g. 'pareto_campaign 2020-12-18_17-38-40'
        """
        data_dir = pathlib.Path(data_dir)

        if dataset_name == "coatings":
            return cls._from_coatings(data_dir / "coatings_2022.csv")

        if dataset_name.startswith("pareto_"):
            stem = dataset_name[len("pareto_"):]
            csv_path = data_dir / f"{stem}.csv"
            return cls._from_pareto(csv_path)

        # Synthetic benchmarks — generic CSV with x0,x1,...,y columns
        if dataset_name in ("hartmann3", "hartmann6", "branin"):
            csv_path = data_dir / f"{dataset_name}.csv"
            return cls._from_synthetic(csv_path)

        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    @classmethod
    def _from_synthetic(cls, csv_path: pathlib.Path) -> "NNOracle":
        """Load a synthetic benchmark oracle from CSV (x0, x1, ..., y columns)."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        y_col = "y"
        x_cols = [c for c in df.columns if c != y_col]
        X = df[x_cols].values.astype(float)
        y = df[y_col].values.astype(float)
        return cls(X, y)

    @classmethod
    def _from_coatings(cls, csv_path: pathlib.Path) -> "NNOracle":
        INPUT_COLS = [
            "concentration_realized", "DMSO_content_realized",
            "combustion_temp_realized", "air_flow_rate_realized",
            "spray_flow_rate_realized", "spray_height_realized",
            "num_passes_realized",
        ]
        OUTPUT_COL = "conductivity_avg"

        df = pd.read_csv(csv_path)
        # Deduplicate: 177 rows → 91 unique experiments (mean over replicates)
        dedup = (
            df.groupby("exp_num")
            .agg({**{c: "mean" for c in INPUT_COLS}, OUTPUT_COL: "mean", "sample": "min"})
            .reset_index()
            .sort_values("sample")
            .reset_index(drop=True)
        )
        X = dedup[INPUT_COLS].values.astype(float)
        y = dedup[OUTPUT_COL].values.astype(float)
        return cls(X, y)

    @classmethod
    def _from_pareto(cls, csv_path: pathlib.Path) -> "NNOracle":
        INPUT_COLS = [
            "x0: fuel to oxidizer ratio", "x1: acac amount",
            "x2: total concentration", "x3: temperature",
        ]
        df = pd.read_csv(csv_path)
        X = df[INPUT_COLS].values.astype(float)

        # Scalarise two objectives: equal-weight sum of z-scored conductance + conductivity
        cond_z = (df["conductance - mean"] - df["conductance - mean"].mean()) / (
            df["conductance - mean"].std() + 1e-12
        )
        cond_z2 = (df["Conductivity - mean"] - df["Conductivity - mean"].mean()) / (
            df["Conductivity - mean"].std() + 1e-12
        )
        y = (cond_z + cond_z2).values.astype(float)
        return cls(X, y)


# Backwards-compatibility alias (old code used GPOracle)
GPOracle = NNOracle
