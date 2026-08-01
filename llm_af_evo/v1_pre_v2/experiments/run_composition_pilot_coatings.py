"""
run_composition_pilot_coatings.py — Approach K gate: "Within-Batch Joint
Composition Pilot", ADA coatings domain. Same lever as
run_composition_pilot.py (does swapping top-k for the baseline's own
novelty-weighted selection close the gap for evolved per-candidate scoring
AFs?) but on the domain where L found a real, replicated gap on top-k
(trust_only +4.8%, 2/3 replicates significant; ehvi_approx ties) — this is
the discriminating test the mAb pilot alone can't provide, since mAb
already ties baseline regardless of selection method.

Reads: if trust_only_novelty on coatings still shows ~+4.8% (same as
trust_only_topk), selection method doesn't modulate the existing gap —
strengthens the "selection is inert, scores are peaked" mechanistic
explanation from the mAb pilot, and closes the approach-K composition
gate with confidence. If trust_only_novelty comes in materially different
(higher, closing the gap toward baseline, or lower), that's the first
positive signal for the composition lever and would need explaining before
ruling K out.

Usage:
    python run_composition_pilot_coatings.py --n_replicates 3 --n_campaigns 20
    python run_composition_pilot_coatings.py --n_replicates 1 --n_campaigns 3  # smoke test
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

from excipient_campaign_mo import run_mo_campaign, make_shared_inits
from strategy_ls_na_egbo import strategy_mo_egbo_novelty
from ada_coatings_oracle import DiscreteADACoatingsOracle
from full_replay import strategy_evolved_af, strategy_evolved_af_novelty
from af_interface import SEED_PROGRAMS
from composition_pilot_common import build_conditions, run_pilot

HERE = pathlib.Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_replicates", type=int, default=3)
    ap.add_argument("--n_campaigns", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--out_path",
                     default=str(HERE / "composition_pilot_coatings_results.json"))
    args = ap.parse_args()

    oracle = DiscreteADACoatingsOracle.build()
    print(f"Oracle: {len(oracle)} real samples, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")

    conditions = build_conditions(strategy_mo_egbo_novelty, strategy_evolved_af,
                                   strategy_evolved_af_novelty, SEED_PROGRAMS)
    run_pilot(conditions, run_mo_campaign, make_shared_inits, strategy_evolved_af,
              strategy_evolved_af_novelty, oracle, args, args.out_path)


if __name__ == "__main__":
    main()
