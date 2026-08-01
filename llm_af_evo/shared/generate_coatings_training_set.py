"""
generate_coatings_training_set.py — coatings analog of generate_training_set.py:
generates the fixed training/held-out set of mo_egbo_novelty campaigns
evolve_af_v2.py's --oracle coatings trains and validates against, in the
same training_logs/{train,heldout}/*.json schema evolve_af_v2.py's
load_training_campaigns / full_replay.reconstruct_oracle(oracle_family=
"coatings") expect (X_init/Y_init/oracle_X_raw/oracle_Y_raw/budget/n_init/
batch_size).

DIFFERENCES from generate_training_set.py, both because coatings is a
single fixed real dataset (ada_coatings_oracle.DiscreteADACoatingsOracle:
253 real measured samples pooled from 4 lab campaigns), not a synthetic
oracle with a redrawable per-seed noise model:
  - No aggregation_tendency sweep — that's a protein-specific dynamics
    knob (excipient_oracle_mo's synthetic model) that has no coatings
    analog. Diversity across training campaigns here comes only from
    different random initial-point draws (make_shared_inits), same as
    run_coatings_generalization.py's own campaign setup.
  - oracle_X_raw/oracle_Y_raw are the SAME real 253-point pool in every
    dumped JSON (nothing is redrawn with fresh noise per seed, since
    there IS no noise model — every value is a real measurement). This
    is expected, not a bug: run_mo_campaign resets each oracle's
    _queried set at the start of every campaign, so every campaign still
    gets access to the full pool regardless of what earlier campaigns
    queried.

COST NOTE: budget=40/n_init=10/batch_size=5 (this file's defaults) is NOT
a fresh guess — it's the exact configuration already run successfully at
n_campaigns=20 in coatings_generalization_results.json (see that file's
per-condition final_hv arrays), so the per-campaign timing risk
run_coatings_generalization.py's own docstring warns about is already
retired for this configuration specifically. Still prints per-campaign
timing here (same as generate_training_set.py doesn't, but
run_coatings_generalization.py does) as a cheap safeguard in case this
run's environment differs from whatever machine produced that file.

Usage:
    python generate_coatings_training_set.py --n_seeds 20 --out_dir training_logs_coatings
    # writes training_logs_coatings/{train,heldout}/*.json
"""

import os

# MUST be set before numpy/scipy/torch import — see evolve_af_v2.py's
# identical block (same rationale, same source: run_coatings
# _generalization.py's directly-verified BLAS non-determinism fix).
# strategy_mo_egbo_novelty here does real GP fitting + optimize_acqf per
# batch, exactly like that file's campaigns, so this training-log
# generation needs the same fix for run-to-run reproducibility.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import pathlib
import sys
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from excipient_campaign_mo import run_mo_campaign, make_shared_inits, pareto_front_of
from strategy_ls_na_egbo import strategy_mo_egbo_novelty

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ada_coatings_oracle import DiscreteADACoatingsOracle

HELDOUT_FRAC = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=20,
                     help="Total campaigns to generate (train+heldout combined) — "
                          "matches the n_campaigns=20 configuration already run "
                          "successfully in coatings_generalization_results.json.")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--rng_seed", type=int, default=42,
                     help="Seed for drawing initial points (make_shared_inits) — "
                          "matches run_coatings_generalization.py's own default seed.")
    ap.add_argument("--out_dir", default=str(pathlib.Path(__file__).parent /
                                              "training_logs_coatings"))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    train_dir = out_dir / "train"
    heldout_dir = out_dir / "heldout"
    train_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    n_heldout = max(1, int(round(args.n_seeds * HELDOUT_FRAC)))
    print(f"{args.n_seeds} seeds ({args.n_seeds - n_heldout} train / {n_heldout} heldout)")

    oracle = DiscreteADACoatingsOracle.build()
    print(f"Oracle: {len(oracle)} real samples, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")

    shared_inits = make_shared_inits(oracle, args.n_seeds, args.n_init, rng_seed=args.rng_seed)

    for seed_idx in range(args.n_seeds):
        X_init, Y_init = shared_inits[seed_idx]

        t0 = time.perf_counter()
        result = run_mo_campaign(
            oracle, X_init, Y_init, args.budget, strategy_mo_egbo_novelty, {},
            batch_size=args.batch_size, seed=seed_idx,
        )
        elapsed = time.perf_counter() - t0
        n_batches = max(1, (args.budget - args.n_init) // args.batch_size)
        print(f"  campaign {seed_idx}: {elapsed:.1f}s total "
              f"({elapsed / n_batches:.2f}s/batch)")

        payload = {
            "seed": seed_idx,
            "condition": "mo_egbo_novelty",
            "oracle": "ada_coatings",
            "budget": args.budget,
            "n_init": args.n_init,
            "batch_size": args.batch_size,
            "X_init": np.asarray(X_init).tolist(),
            "Y_init": np.asarray(Y_init).tolist(),
            "hv_trajectory": result["hv_trajectory"],
            "decisions": result["decisions"],
            "final_pf_size": len(pareto_front_of(
                result["Y_obs"], directions=oracle.objective_directions())),
            "oracle_X_raw": oracle._X_raw.tolist(),
            "oracle_Y_raw": oracle._Y_raw.tolist(),
        }

        split_dir = heldout_dir if seed_idx < n_heldout else train_dir
        fname = f"coatings_seed{seed_idx:03d}.json"
        with open(split_dir / fname, "w") as f:
            json.dump(payload, f)

    n_train = len(list(train_dir.glob("*.json")))
    n_heldout_total = len(list(heldout_dir.glob("*.json")))
    print(f"\nTraining set: {n_train} campaigns in {train_dir}")
    print(f"Held-out set: {n_heldout_total} campaigns in {heldout_dir}")


if __name__ == "__main__":
    main()
