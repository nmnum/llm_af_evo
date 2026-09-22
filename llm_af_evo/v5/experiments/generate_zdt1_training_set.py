"""
generate_zdt1_training_set.py — ZDT1 sibling of generate_dtlz2_training_set.py.
See that file's own module docstring for the full design rationale (one
shared oracle pool, diversity from init points only, no noise config to
worry about); everything there applies identically here.

WHY a second synthetic benchmark, not just DTLZ2: v5's champion (run1,
gen148_jitter1) showed a small positive margin on a DTLZ2 held-out check
(+0.28%, n=5 — too small to trust on its own). Before treating that as a
real "generalizes to synthetic benchmarks" finding, it needs to hold up on
a SECOND, independently-designed synthetic benchmark too — DTLZ2 and ZDT1
are both synthetic, but they're structurally different problems (DTLZ2:
concave front, uniform curvature, n_obj>=3 possible; ZDT1: convex front,
2-objective, different variable roles for convergence vs. diversity) — a
champion that only works on DTLZ2's specific front geometry and not ZDT1's
would still be a form of overfitting, just to "DTLZ2-shaped fronts"
instead of "tunable/mAb/coatings-shaped fronts". Passing both is a
meaningfully stronger claim than passing either alone.

Usage:
    python generate_zdt1_training_set.py --n_seeds 20 --heldout_frac 1.0 \\
        --out_dir training_logs_zdt1
    # writes training_logs_zdt1/heldout/*.json (train/ empty at heldout_frac=1.0)
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

DEFAULT_D = 6  # matches full_replay._ORACLE_SHAPE_EXPECTATIONS["zdt1"]["feature_dim"]
DEFAULT_POOL_SIZE = 500  # matches generate_dtlz2_training_set.py's default,
                          # itself matching run_dtlz2_generalization.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=20,
                     help="Total campaigns to generate (train+heldout combined).")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--pool_size", type=int, default=DEFAULT_POOL_SIZE)
    ap.add_argument("--d", type=int, default=DEFAULT_D,
                     help="Input dimensionality — must match "
                          "full_replay._ORACLE_SHAPE_EXPECTATIONS['zdt1']"
                          "['feature_dim'] if that dict isn't updated to match.")
    ap.add_argument("--zdt1_seed", type=int, default=0,
                     help="Seed for the ONE shared oracle pool build (X_raw "
                          "draw), reused across every campaign — see module "
                          "docstring's design note. NOT a per-campaign seed.")
    ap.add_argument("--rng_seed", type=int, default=42,
                     help="Seed for drawing initial points (make_shared_inits).")
    ap.add_argument("--heldout_frac", type=float, default=HELDOUT_FRAC,
                     help="Fraction of --n_seeds campaigns written to heldout/ "
                          "rather than train/. ZDT1, like DTLZ2, is a "
                          "never-trained-on validation domain for v5 — pass 1.0 "
                          "to put every generated campaign into heldout/ instead "
                          "of wasting most of --n_seeds on an unused train/ split.")
    ap.add_argument("--out_dir", default=str(pathlib.Path(__file__).parent /
                                              "training_logs_zdt1"))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    train_dir = out_dir / "train"
    heldout_dir = out_dir / "heldout"
    train_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    n_heldout = max(1, int(round(args.n_seeds * args.heldout_frac)))
    print(f"{args.n_seeds} seeds ({args.n_seeds - n_heldout} train / {n_heldout} heldout)")
    print(f"config: d={args.d} pool_size={args.pool_size}")

    # ONE shared oracle for every campaign — see module docstring for why
    # NOT a fresh oracle per campaign.
    oracle = DiscreteSyntheticMOOracle.build_zdt1(
        d=args.d, pool_size=args.pool_size, seed=args.zdt1_seed)
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
            "oracle": "zdt1",
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
            # No oracle_Y_true / zdt1_* config needed — DiscreteSyntheticMOOracle
            # is fully deterministic, so reconstruct_oracle(oracle_family="zdt1")
            # only needs X_raw/Y_raw (see its own docstring).
        }

        split_dir = heldout_dir if seed_idx < n_heldout else train_dir
        fname = f"zdt1_seed{seed_idx:03d}.json"
        with open(split_dir / fname, "w") as f:
            json.dump(payload, f)

    n_train = len(list(train_dir.glob("*.json")))
    n_heldout_total = len(list(heldout_dir.glob("*.json")))
    print(f"\nTraining set: {n_train} campaigns in {train_dir}")
    print(f"Held-out set: {n_heldout_total} campaigns in {heldout_dir}")


if __name__ == "__main__":
    main()
