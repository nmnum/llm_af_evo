"""
generate_dtlz2_training_set.py — DTLZ2 analog of
generate_tunable_training_set.py / generate_coatings_training_set.py:
generates the fixed training/held-out set of mo_egbo_novelty campaigns
that v5's multi-domain evolution holds out entirely (see the af-evolution
branch conversation's multi-domain-fitness design: DTLZ2 is validation-only,
never trained on, precisely so it can catch a formula that's overfit to
tunable/mAb/coatings' shared structure rather than the general task).

Unlike generate_tunable_training_set.py, there is no noise configuration
here at all: DiscreteSyntheticMOOracle.build_dtlz2 is fully deterministic
(Y_raw = _dtlz2(X_raw) directly — see synthetic_mo_oracle.py). So "campaign
diversity" has only ONE source, not two: different initial-point draws
across campaigns, from ONE shared oracle pool built once. This matches
generate_tunable_training_set.py/generate_coatings_training_set.py's own
established pattern (one fixed pool, diversity from init points only) —
NOT a fresh oracle per campaign, which that file's own history shows
inflates cross-campaign CV from a validated ~2-3% to ~23% on the tunable
domain, by introducing pool-quality variance that has nothing to do with
GP-fit/strategy noise. The same risk applies here: DTLZ2's pool is drawn
uniformly in [0,1]^d, and re-drawing it per campaign would let a lucky/
unlucky draw's proximity to the true front dominate the signal instead of
genuine AF-quality differences.

reconstruct_oracle(oracle_family="dtlz2") (full_replay.py) is deliberately
simple compared to its "tunable" branch: no oracle_Y_true, no tunable_*
config fields needed in the payload, since there's no noise realization to
avoid re-drawing. Only oracle_X_raw/oracle_Y_raw are required, exactly
like the "coatings" branch's real-fixed-data pattern.

Usage:
    python generate_dtlz2_training_set.py --n_seeds 20 --out_dir training_logs_dtlz2
    # writes training_logs_dtlz2/{train,heldout}/*.json
"""

import os

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

_LLM_AF_EVO = pathlib.Path(__file__).resolve().parent
while _LLM_AF_EVO.name != "llm_af_evo":
    _LLM_AF_EVO = _LLM_AF_EVO.parent
_ROOT = _LLM_AF_EVO.parent
for _p in (
    _ROOT,
    _LLM_AF_EVO / "shared",
    _LLM_AF_EVO / "v1_pre_v2" / "src",
    _LLM_AF_EVO / "v1_pre_v2" / "experiments",
    _LLM_AF_EVO / "v2" / "src",
    _LLM_AF_EVO / "v2" / "experiments",
):
    sys.path.insert(0, str(_p))

from excipient_campaign_mo import run_mo_campaign, make_shared_inits, pareto_front_of
from strategy_ls_na_egbo import strategy_mo_egbo_novelty

from synthetic_mo_oracle import DiscreteSyntheticMOOracle

HELDOUT_FRAC = 0.25

# Matches run_dtlz2_generalization.py's own defaults (the script this
# project already used for the front-range-normalisation audit and the
# v3-champion transfer check), so this training set is directly comparable
# to those prior results rather than a new, unvalidated configuration.
DEFAULT_N_OBJ = 3
DEFAULT_K = 4  # feature_dim = n_obj - 1 + k = 6, matching
               # full_replay._ORACLE_SHAPE_EXPECTATIONS["dtlz2"]["feature_dim"]
DEFAULT_POOL_SIZE = 500  # matches run_dtlz2_generalization.py's oracle,
                          # which reported "500 pool points" at these defaults


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=20,
                     help="Total campaigns to generate (train+heldout combined).")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--pool_size", type=int, default=DEFAULT_POOL_SIZE)
    ap.add_argument("--n_obj", type=int, default=DEFAULT_N_OBJ)
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                     help="DTLZ2's k parameter — feature_dim = n_obj - 1 + k. "
                          "Must match full_replay._ORACLE_SHAPE_EXPECTATIONS "
                          "['dtlz2']['feature_dim'] if that dict isn't updated "
                          "to match a non-default choice here.")
    ap.add_argument("--dtlz2_seed", type=int, default=0,
                     help="Seed for the ONE shared oracle pool build (X_raw "
                          "draw), reused across every campaign — see module "
                          "docstring's design note. NOT a per-campaign seed.")
    ap.add_argument("--rng_seed", type=int, default=42,
                     help="Seed for drawing initial points (make_shared_inits).")
    ap.add_argument("--heldout_frac", type=float, default=HELDOUT_FRAC,
                     help="Fraction of --n_seeds campaigns written to heldout/ "
                          "rather than train/. Defaults to the same 0.25 every "
                          "other training-set generator in this project uses, "
                          "but v5 never actually trains on DTLZ2 (it's held out "
                          "entirely as a never-trained-on validation check — see "
                          "evolve_af_v5.py's module docstring) — pass 1.0 to put "
                          "every generated campaign into heldout/ instead of "
                          "wasting most of --n_seeds on an unused train/ split.")
    ap.add_argument("--out_dir", default=str(pathlib.Path(__file__).parent /
                                              "training_logs_dtlz2"))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    train_dir = out_dir / "train"
    heldout_dir = out_dir / "heldout"
    train_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    n_heldout = max(1, int(round(args.n_seeds * args.heldout_frac)))
    print(f"{args.n_seeds} seeds ({args.n_seeds - n_heldout} train / {n_heldout} heldout)")
    print(f"config: n_obj={args.n_obj} k={args.k} pool_size={args.pool_size} "
          f"(feature_dim={args.n_obj - 1 + args.k})")

    # ONE shared oracle for every campaign — see module docstring for why
    # NOT a fresh oracle per campaign.
    oracle = DiscreteSyntheticMOOracle.build_dtlz2(
        n_obj=args.n_obj, k=args.k, pool_size=args.pool_size, seed=args.dtlz2_seed)
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
            "oracle": "dtlz2",
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
            # No oracle_Y_true / dtlz2_* config needed — DiscreteSyntheticMOOracle
            # is fully deterministic, so reconstruct_oracle(oracle_family="dtlz2")
            # only needs X_raw/Y_raw (see its own docstring).
        }

        split_dir = heldout_dir if seed_idx < n_heldout else train_dir
        fname = f"dtlz2_seed{seed_idx:03d}.json"
        with open(split_dir / fname, "w") as f:
            json.dump(payload, f)

    n_train = len(list(train_dir.glob("*.json")))
    n_heldout_total = len(list(heldout_dir.glob("*.json")))
    print(f"\nTraining set: {n_train} campaigns in {train_dir}")
    print(f"Held-out set: {n_heldout_total} campaigns in {heldout_dir}")


if __name__ == "__main__":
    main()
