"""
generate_noisy_synthetic_training_set.py — v6's training-set generator for
the noisy-synthetic substrate (see v6/README.md's "Substrate" decision).

One parameterised script covering all four v6 synthetic domains (ZDT1,
ZDT3, DTLZ2 at n_obj=3 for training, DTLZ2 at n_obj=5 for held-out),
rather than v5's one-file-per-domain pattern
(generate_dtlz2_training_set.py/generate_zdt1_training_set.py) — v5's
domains differed only in WHICH builder to call; v6's domains differ in
WHICH BUILDER *and* WHAT NOISE LEVEL, and the whole point of this
substrate is that noise_cv is a first-class dial (see v6/README.md: "the
noise level becomes a tunable parameter — you could even sweep it"), so
parameterising it directly here rather than hardcoding one level per
file matches that design more directly.

Ground-truth generation logic (oracle pool build, shared inits, campaign
running via strategy_mo_egbo_novelty) is otherwise IDENTICAL to
generate_dtlz2_training_set.py's own pattern — ONE shared oracle built
once, campaign diversity from init-point draws only (see that file's
own docstring for why NOT a fresh oracle per campaign: pool-quality
variance would dominate over genuine AF-quality differences).

v6-specific additions to the payload (beyond v5's dtlz2/zdt1 fields):
  - "noise_cv": the multiplicative observation-noise CV baked into this
    domain's oracle (see synthetic_mo_oracle.py's noise model docstring).
    0.0 reproduces v5's noiseless behaviour exactly.
  - "noise_seed": seeds the oracle's noise RNG — see
    full_replay.reconstruct_oracle's dtlz2/zdt1/zdt3 branch, which reads
    both these keys with 0.0/42 fallbacks so pre-v6 logs (neither key
    present) replay exactly as before.

Usage (the exact four domain/noise assignments from v6/README.md):
    python generate_noisy_synthetic_training_set.py --domain zdt1 --noise_cv 0.15 \\
        --out_dir training_logs_zdt1_n15
    python generate_noisy_synthetic_training_set.py --domain dtlz2 --n_obj 3 --noise_cv 0.30 \\
        --out_dir training_logs_dtlz2_3obj_n30
    python generate_noisy_synthetic_training_set.py --domain zdt3 --noise_cv 0.50 \\
        --out_dir training_logs_zdt3_n50
    python generate_noisy_synthetic_training_set.py --domain dtlz2 --n_obj 5 --noise_cv 0.35 \\
        --heldout_frac 1.0 --out_dir training_logs_dtlz2_5obj_n35_heldout
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
DEFAULT_POOL_SIZE = 500
DEFAULT_K = 4  # DTLZ2 only: feature_dim = n_obj - 1 + k


def build_oracle(domain: str, n_obj: int, k: int, pool_size: int,
                  data_seed: int, noise_cv: float, noise_seed: int):
    if domain == "zdt1":
        return DiscreteSyntheticMOOracle.build_zdt1(
            pool_size=pool_size, seed=data_seed, noise_cv=noise_cv, noise_seed=noise_seed)
    if domain == "zdt3":
        return DiscreteSyntheticMOOracle.build_zdt3(
            pool_size=pool_size, seed=data_seed, noise_cv=noise_cv, noise_seed=noise_seed)
    if domain == "dtlz2":
        return DiscreteSyntheticMOOracle.build_dtlz2(
            n_obj=n_obj, k=k, pool_size=pool_size, seed=data_seed,
            noise_cv=noise_cv, noise_seed=noise_seed)
    raise ValueError(f"unknown domain {domain!r} — must be zdt1/zdt3/dtlz2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["zdt1", "zdt3", "dtlz2"])
    ap.add_argument("--noise_cv", type=float, required=True,
                     help="Multiplicative Gaussian observation-noise CV — "
                          "0.0 reproduces v5's noiseless behaviour exactly. "
                          "See v6/README.md for the exact per-domain "
                          "assignment (ZDT1@0.15, DTLZ2-3@0.30, ZDT3@0.50, "
                          "DTLZ2-5@0.35).")
    ap.add_argument("--noise_seed", type=int, default=42)
    ap.add_argument("--n_obj", type=int, default=3,
                     help="DTLZ2 only — ignored for zdt1/zdt3 (always "
                          "2-objective). Use 3 for the training domain, 5 "
                          "for the held-out-synthetic domain.")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                     help="DTLZ2 only — feature_dim = n_obj - 1 + k.")
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--pool_size", type=int, default=DEFAULT_POOL_SIZE)
    ap.add_argument("--data_seed", type=int, default=0,
                     help="Seed for the ONE shared oracle pool build (X_raw "
                          "draw) — NOT the noise seed, and NOT a per-campaign "
                          "seed. See module docstring's design note.")
    ap.add_argument("--rng_seed", type=int, default=42,
                     help="Seed for drawing initial points (make_shared_inits).")
    ap.add_argument("--heldout_frac", type=float, default=HELDOUT_FRAC,
                     help="Fraction of --n_seeds written to heldout/ rather "
                          "than train/. Pass 1.0 for a domain that's "
                          "validation-only (e.g. DTLZ2-5, never trained on — "
                          "see v6/README.md).")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    train_dir = out_dir / "train"
    heldout_dir = out_dir / "heldout"
    train_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    n_heldout = max(1, int(round(args.n_seeds * args.heldout_frac)))
    print(f"domain={args.domain} noise_cv={args.noise_cv} "
          f"n_obj={args.n_obj if args.domain == 'dtlz2' else 2}")
    print(f"{args.n_seeds} seeds ({args.n_seeds - n_heldout} train / {n_heldout} heldout)")

    oracle = build_oracle(args.domain, args.n_obj, args.k, args.pool_size,
                           args.data_seed, args.noise_cv, args.noise_seed)
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
            "oracle": args.domain,
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
            # v6 additions — see module docstring and
            # full_replay.reconstruct_oracle's dtlz2/zdt1/zdt3 branch.
            "noise_cv": args.noise_cv,
            "noise_seed": args.noise_seed,
        }

        split_dir = heldout_dir if seed_idx < n_heldout else train_dir
        fname = f"{args.domain}_seed{seed_idx:03d}.json"
        with open(split_dir / fname, "w") as f:
            json.dump(payload, f)

    n_train = len(list(train_dir.glob("*.json")))
    n_heldout_total = len(list(heldout_dir.glob("*.json")))
    print(f"\nTraining set: {n_train} campaigns in {train_dir}")
    print(f"Held-out set: {n_heldout_total} campaigns in {heldout_dir}")


if __name__ == "__main__":
    main()
