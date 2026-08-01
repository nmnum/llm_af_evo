"""
ada_coatings_oracle.py — real-data discrete-pool oracle for MacLeod et al.
2022 ("Self-driving laboratories can advance the Pareto front for thin-film
materials", Nature Communications, github.com/berlinguette/ada).

Pools all 4 real experimental campaigns (253 total samples) into one fixed
discrete pool, queried by nearest-neighbour snap — same architecture as
excipient_oracle_mo.DiscreteMOExcipientOracle, but with ZERO fitting or
extrapolation: every query returns an ACTUAL measured value from the real
dataset, never a predicted/synthetic one. This is deliberate, not just
convenient — an earlier phase of this project (synthetic_coatings_oracle.py)
documents that a naive GP surrogate fit to this same lab's coatings data
produced a 193x empirical-max extrapolation artifact; snap-to-real-data
sidesteps that failure mode entirely by construction, since nothing is
ever extrapolated.

Objectives — REVISED from the original 3-objective version. The raw data
has three co-measured conductivity-family columns (conductance, XRF-
normalized conductance, conductivity), but they do not constitute a real
multi-objective problem: xrf_conductance and conductivity are correlated
at r=1.0000 (essentially the same measurement up to linear rescaling —
verified directly against the pooled 253-sample data), and even the third
(raw conductance) only reaches r~0.50 with the other two. The resulting
"Pareto front" over all three was 5/253 points (2.0%) — a near-degenerate
single-objective problem in a 3-objective disguise, confirmed BEFORE this
revision, not assumed.

The objectives used here instead — conductivity (maximize) and
conductance-std (minimize, a within-sample uniformity proxy: higher
position-to-position variability in conductance means a less uniform
film) — are genuinely independent (r=0.0032 across the same 253 samples)
and produce a real trade-off front (79/253 points, 31.2%). This is the
oracle actually used for the coatings generalization test; the original
3-objective version is not used.

  conductivity      (S/m, maximize)
  conductance_std   (Siemens, minimize — uniformity proxy)

Inputs (4D, continuous, real experimental ranges — NOT normalised to
[0,1], unlike the excipient oracle's categorical encoding; bounds() below
reflects the actual observed range, since there's no natural [0,1] scale
for "fuel:oxidizer ratio" etc.):
  x0  fuel to oxidizer ratio
  x1  acac amount (glycine <-> acetylacetone composition)
  x2  total precursor concentration (g/mL)
  x3  annealing temperature (Celsius)

Data provenance: coatings_data/*.csv, downloaded from
github.com/berlinguette/ada's "2021_01 ... Pareto front ..." folder (Git
LFS objects, resolved via media.githubusercontent.com since raw.
githubusercontent.com only serves LFS pointer files for this repo).
"""

import csv
import pathlib
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).parent
DATA_DIR = HERE / "coatings_data"

FEATURE_DIM = 4
OBJECTIVE_NAMES = ["conductivity", "conductance_std"]
OBJECTIVE_DIRECTIONS = ["max", "min"]

_X_COLS = ["x0: fuel to oxidizer ratio", "x1: acac amount",
           "x2: total concentration", "x3: temperature"]
_Y_COLS = ["Conductivity - mean", "conductance - std"]


def load_pooled_data(data_dir: pathlib.Path = DATA_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """Pool every campaign CSV in data_dir into one (X_raw, Y_raw) pair."""
    files = sorted(pathlib.Path(data_dir).glob("campaign_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No campaign CSVs found in {data_dir} — expected files like "
            f"'campaign_2020-12-18_17-38-40.csv' (downloaded from "
            f"github.com/berlinguette/ada).")

    X_rows, Y_rows = [], []
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                X_rows.append([float(row[c]) for c in _X_COLS])
                Y_rows.append([float(row[c]) for c in _Y_COLS])

    return np.array(X_rows), np.array(Y_rows)


class DiscreteADACoatingsOracle:
    """
    Real-data discrete pool oracle, interface-compatible with
    excipient_oracle_mo.DiscreteMOExcipientOracle (bounds(),
    objective_directions(), objective_names(), query_mo(), and the
    _X_raw/_Y_raw/_scaler/_queried attributes the shared campaign-running
    code — run_mo_campaign, strategy_mo_egbo_novelty, strategy_evolved_af —
    already expects).
    """

    def __init__(self, X_raw: np.ndarray, Y_raw: np.ndarray):
        self._X_raw = X_raw
        self._Y_raw = Y_raw
        self._scaler = StandardScaler().fit(X_raw)
        self._queried = set()
        self._pareto_idx_cache = None

    @classmethod
    def build(cls, data_dir: pathlib.Path = DATA_DIR) -> "DiscreteADACoatingsOracle":
        X_raw, Y_raw = load_pooled_data(data_dir)
        return cls(X_raw, Y_raw)

    def __len__(self) -> int:
        return len(self._X_raw)

    def bounds(self) -> np.ndarray:
        """
        Observed min/max per input dimension — there's no natural [0,1]
        scale for these physical units (e.g. fuel:oxidizer ratio observed
        up to ~2.5, well past the "0 to 1" nominal description in the
        paper's README), so bounds reflect the actual explored range.
        """
        return np.column_stack([self._X_raw.min(axis=0), self._X_raw.max(axis=0)])

    def objective_directions(self) -> List[str]:
        return list(OBJECTIVE_DIRECTIONS)

    def objective_names(self) -> List[str]:
        return list(OBJECTIVE_NAMES)

    def query_mo(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Nearest-neighbour lookup against the real pooled dataset. Returns
        (objective_vector, pool_index) — the objective vector is always a
        REAL measured value, never predicted."""
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
    oracle = DiscreteADACoatingsOracle.build()
    print(f"Pooled ADA coatings oracle: {len(oracle)} real samples, "
          f"{FEATURE_DIM}D inputs, {len(OBJECTIVE_NAMES)} objectives "
          f"{OBJECTIVE_NAMES}")
    print(f"Input bounds:\n{oracle.bounds()}")
    print(f"Objective ranges:\n"
          f"{np.column_stack([oracle._Y_raw.min(axis=0), oracle._Y_raw.max(axis=0)])}")
