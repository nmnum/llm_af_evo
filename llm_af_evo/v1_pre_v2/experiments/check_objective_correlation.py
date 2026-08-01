"""
check_objective_correlation.py — cheap, data-only prerequisite check
before spending real-domain compute on DA-COREG: its entire value
proposition is sharing structure across CORRELATED objectives (a
coregionalized multi-task GP has nothing to exploit if the objectives are
independent). Coatings' two objectives were deliberately rebuilt earlier
in this project to have near-zero correlation (r=0.0032) specifically to
make it a genuine multi-objective trade-off — which means DA-COREG likely
has the LEAST to offer there, structurally, regardless of implementation
quality. mAb's Tm/kD/viscosity correlation was never checked. This script
reports both, from the same real/synthetic data every other pilot in this
project uses, so DA-COREG's real-domain runs can be sequenced by where the
mechanism has a chance of mattering rather than run blind.

No campaign compute — just np.corrcoef over each oracle's full raw
objective matrix.

Usage:
    python check_objective_correlation.py
"""

import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from excipient_oracle_mo import MultiObjectiveExcipientOracle
from ada_coatings_oracle import DiscreteADACoatingsOracle
from synthetic_mo_oracle import DiscreteSyntheticMOOracle

PROTEIN = "mAb_aggregation"


def report(name, Y_raw, objective_names):
    print(f"\n{name}: {Y_raw.shape[0]} points, objectives {objective_names}")
    corr = np.corrcoef(Y_raw.T)
    n = len(objective_names)
    max_abs_r = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = corr[i, j]
            max_abs_r = max(max_abs_r, abs(r))
            print(f"  corr({objective_names[i]}, {objective_names[j]}) = {r:+.4f}")
    verdict = ("STRUCTURE TO SHARE — DA-COREG has something to exploit here"
               if max_abs_r > 0.15 else
               "NEAR-INDEPENDENT — DA-COREG likely has little/nothing to exploit here")
    print(f"  max |r| = {max_abs_r:.4f}  ->  {verdict}")


if __name__ == "__main__":
    mab_full = MultiObjectiveExcipientOracle(
        protein=PROTEIN, tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10, seed=42)
    mab = mab_full.make_discrete_oracle(n_samples=500, seed=42)
    report("mAb (excipient formulation)", mab._Y_raw, mab.objective_names())

    coatings = DiscreteADACoatingsOracle.build()
    report("ADA coatings", coatings._Y_raw, coatings.objective_names())

    dtlz2 = DiscreteSyntheticMOOracle.build_dtlz2()
    report("DTLZ2 (synthetic, engineered shared g(x) term across all objectives)",
           dtlz2._Y_raw, dtlz2.objective_names())

    zdt1 = DiscreteSyntheticMOOracle.build_zdt1()
    report("ZDT1 (synthetic, f2 depends on f1 through shared g(x))",
           zdt1._Y_raw, zdt1.objective_names())
