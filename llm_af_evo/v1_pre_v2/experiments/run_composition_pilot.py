"""
run_composition_pilot.py — Approach K gate: "Within-Batch Joint Composition
Pilot", mAb excipient domain. Tests whether L's negative result (evolved
per-candidate scoring ties, doesn't beat, the baseline) was partly an
artifact of L's fixed top-k batch-selection step, rather than of
per-candidate scoring quality itself. L always paired evolved scores with
pure top-k (select_batch); this pilot pairs the SAME evolved scores with
the baseline's own novelty-weighted greedy selection
(novelty_aware_select_vectorised, real defaults w_acq=0.9/w_nov=0.1) via
strategy_evolved_af_novelty, and compares against top-k selection on the
identical scores.

Conditions (flat CONDITIONS dict, see composition_pilot_common.build_conditions):
  mo_egbo_novelty        — baseline: qLogNEHVI scoring + novelty selection (unmodified)
  trust_only_topk        — evolved scoring (trust_only) + top-k selection (existing L design)
  trust_only_novelty     — evolved scoring (trust_only) + novelty selection (this pilot's lever)
  ehvi_approx_topk       — evolved scoring (ehvi_approx) + top-k selection
  ehvi_approx_novelty    — evolved scoring (ehvi_approx) + novelty selection

Decision gate (pre-committed): if trust_only_novelty (or ehvi_approx_novelty)
beats trust_only_topk (or ehvi_approx_topk) AND matches/beats mo_egbo_novelty,
that's evidence the composition/selection step — not per-candidate scoring —
is where L's gap actually lives. IMPORTANT CAVEAT (per the mAb result already
observed): mAb is the domain where L already ties baseline (trust_only +2.4%
p=0.50, ehvi_approx +1.0% p=0.70) — a null here is consistent with "selection
method is inert" but is also consistent with "there was no gap to close in
the first place." The discriminating test is the SAME pilot on coatings
(run_composition_pilot_coatings.py), where a real gap exists (trust_only
+4.8%, 2/3 replicates significant on top-k) — only a null there, alongside
this mAb null, closes the gate with confidence.

Usage:
    python run_composition_pilot.py --n_replicates 3 --n_campaigns 20
    python run_composition_pilot.py --n_replicates 1 --n_campaigns 3   # timing smoke test
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from excipient_oracle_mo import MultiObjectiveExcipientOracle
from excipient_campaign_mo import run_mo_campaign, make_shared_inits
from strategy_ls_na_egbo import strategy_mo_egbo_novelty
from full_replay import strategy_evolved_af, strategy_evolved_af_novelty
from af_interface import SEED_PROGRAMS
from composition_pilot_common import build_conditions, run_pilot

HERE = pathlib.Path(__file__).parent
PROTEIN = "mAb_aggregation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_replicates", type=int, default=3)
    ap.add_argument("--n_campaigns", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--out_path", default=str(HERE / "composition_pilot_results.json"))
    args = ap.parse_args()

    oracle_full = MultiObjectiveExcipientOracle(
        protein=PROTEIN, tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10, seed=42)
    oracle = oracle_full.make_discrete_oracle(n_samples=500, seed=42)
    print(f"Oracle: {len(oracle)} pool points, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")

    conditions = build_conditions(strategy_mo_egbo_novelty, strategy_evolved_af,
                                   strategy_evolved_af_novelty, SEED_PROGRAMS)
    run_pilot(conditions, run_mo_campaign, make_shared_inits, strategy_evolved_af,
              strategy_evolved_af_novelty, oracle, args, args.out_path)


if __name__ == "__main__":
    main()
