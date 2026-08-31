"""
generate_tunable_training_set.py — tunable-synthetic-domain analog of
generate_coatings_training_set.py: generates the fixed training/held-out
set of mo_egbo_novelty campaigns evolve_af_v3.py's --oracle tunable trains
and validates against, in the training_logs/{train,heldout}/*.json schema
evolve_af_v3.py's load_training_campaigns / full_replay.reconstruct_oracle
(oracle_family="tunable") expect.

DEFAULTS ARE NOT A FRESH GUESS: plateau_sharpness=5.0, noise_level=0.08,
noise_mode="proportional", scale2=3.0 are exactly the configuration
tunable_domain_generalization_results.json already measured at CV~1.5-2.9%
across 20 campaigns (vs. excipient/mAb's ~50% CV) — see
llm_af_evo/v3/README.md's audit note and tunable_synthetic_oracle.py's own
docstring for why scale2 specifically should not be casually increased
(mu_sum dominance risk above ~3-5x per that file's dominance_ratio sweep).

DIFFERENCE from generate_coatings_training_set.py: coatings reuses ONE
fixed real oracle (253 real measured points) across every campaign, so
diversity comes only from different initial-point draws. This domain is
synthetic and redrawable, so EVERY campaign gets its own freshly-built
TunableSyntheticMOOracle (fresh --pool_size random X_raw, fresh noise
draw) at seed=seed_idx — genuinely different pools per campaign, not
resampled subsets of one fixed dataset. Both oracle_Y_raw (noisy,
observed) AND oracle_Y_true (noiseless ground truth) are dumped per
campaign, because full_replay.reconstruct_oracle(oracle_family="tunable")
needs Y_true to reconstruct via TunableSyntheticMOOracle.
from_fixed_realization WITHOUT re-drawing noise on top of an already-noisy
Y_raw (see that constructor's docstring — this is the fix this session
made specifically so this generator's output replays deterministically).

CAMPAIGN COUNT: defaults to n_seeds=24, much smaller than excipient's
n_campaigns~75 need. At this domain's measured CV~2.8%, SE(mean_margin) at
n=8 is already ~1% (see the af-evolution branch conversation's SE table) —
close to the margin gaps this project targets, unlike excipient where even
n=75 leaves SE~5x the target gap. 24 (18 train / 6 heldout at the same
0.25 HELDOUT_FRAC as coatings) is a reasonable starting size, not a
load-bearing constant — increase via --n_seeds if a real run's own
bootstrap CI says it needs more.

Usage:
    python generate_tunable_training_set.py --n_seeds 24 --out_dir training_logs_tunable
    # writes training_logs_tunable/{train,heldout}/*.json
"""

import os

# MUST be set before numpy/scipy/torch import — same rationale as
# generate_coatings_training_set.py's identical block: strategy_mo_egbo_
# novelty does real GP fitting + optimize_acqf per batch here too.
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
    _LLM_AF_EVO / "v3" / "src",
    _LLM_AF_EVO / "v3" / "experiments",
):
    sys.path.insert(0, str(_p))

from excipient_campaign_mo import run_mo_campaign, make_shared_inits, pareto_front_of
from strategy_ls_na_egbo import strategy_mo_egbo_novelty

from tunable_synthetic_oracle import TunableSyntheticMOOracle

HELDOUT_FRAC = 0.25

# Validated defaults — see this file's module docstring. Do not change
# scale2 without re-running sweep_tunable_domain.py's dominance_ratio
# check first (tunable_synthetic_oracle.py's own docstring warning).
DEFAULT_D = 6
DEFAULT_POOL_SIZE = 256
DEFAULT_SCALE1 = 1.0
DEFAULT_SCALE2 = 3.0
DEFAULT_PLATEAU_SHARPNESS = 5.0
DEFAULT_NOISE_LEVEL = 0.08
DEFAULT_NOISE_MODE = "proportional"
DEFAULT_BOUNDARY_GAIN = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=24,
                     help="Total campaigns to generate (train+heldout combined) — "
                          "see module docstring's CAMPAIGN COUNT note for why this "
                          "is much smaller than excipient's ~75.")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--pool_size", type=int, default=DEFAULT_POOL_SIZE)
    ap.add_argument("--d", type=int, default=DEFAULT_D,
                     help="Input dimensionality — must match "
                          "full_replay._ORACLE_SHAPE_EXPECTATIONS['tunable']['feature_dim'] "
                          "if that dict isn't updated to match.")
    ap.add_argument("--scale1", type=float, default=DEFAULT_SCALE1)
    ap.add_argument("--scale2", type=float, default=DEFAULT_SCALE2)
    ap.add_argument("--plateau_sharpness", type=float, default=DEFAULT_PLATEAU_SHARPNESS)
    ap.add_argument("--noise_level", type=float, default=DEFAULT_NOISE_LEVEL)
    ap.add_argument("--noise_mode", default=DEFAULT_NOISE_MODE,
                     choices=["homoscedastic", "proportional", "input_dependent"])
    ap.add_argument("--boundary_gain", type=float, default=DEFAULT_BOUNDARY_GAIN)
    ap.add_argument("--rng_seed", type=int, default=42,
                     help="Seed for drawing initial points (make_shared_inits).")
    ap.add_argument("--out_dir", default=str(pathlib.Path(__file__).parent /
                                              "training_logs_tunable"))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    train_dir = out_dir / "train"
    heldout_dir = out_dir / "heldout"
    train_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    n_heldout = max(1, int(round(args.n_seeds * HELDOUT_FRAC)))
    print(f"{args.n_seeds} seeds ({args.n_seeds - n_heldout} train / {n_heldout} heldout)")
    print(f"config: d={args.d} pool_size={args.pool_size} scale1={args.scale1} "
          f"scale2={args.scale2} plateau_sharpness={args.plateau_sharpness} "
          f"noise_level={args.noise_level} noise_mode={args.noise_mode!r} "
          f"boundary_gain={args.boundary_gain}")

    for seed_idx in range(args.n_seeds):
        # Fresh oracle per campaign — see module docstring's DIFFERENCE
        # note. seed=seed_idx drives BOTH the random X_raw pool draw and
        # the noise realization (see TunableSyntheticMOOracle.build).
        oracle = TunableSyntheticMOOracle.build(
            d=args.d, pool_size=args.pool_size,
            scale1=args.scale1, scale2=args.scale2,
            plateau_sharpness=args.plateau_sharpness,
            noise_level=args.noise_level, noise_mode=args.noise_mode,
            boundary_gain=args.boundary_gain, seed=seed_idx,
        )

        shared_inits = make_shared_inits(oracle, 1, args.n_init, rng_seed=args.rng_seed + seed_idx)
        X_init, Y_init = shared_inits[0]

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
            "oracle": "tunable",
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
            # noiseless ground truth — REQUIRED by full_replay.reconstruct_oracle's
            # "tunable" branch (via from_fixed_realization) so replay doesn't
            # re-draw noise on top of this already-noisy oracle_Y_raw.
            "oracle_Y_true": oracle._Y_true.tolist(),
            # tunable_* config — REQUIRED by full_replay.reconstruct_oracle's
            # "tunable" branch (see its _TUNABLE_DEFAULTS comment) to
            # reconstruct THIS campaign's exact oracle, not generic defaults.
            "tunable_scale1": args.scale1,
            "tunable_scale2": args.scale2,
            "tunable_noise_level": args.noise_level,
            "tunable_noise_mode": args.noise_mode,
            "tunable_boundary_gain": args.boundary_gain,
            "tunable_seed": seed_idx,
        }

        split_dir = heldout_dir if seed_idx < n_heldout else train_dir
        fname = f"tunable_seed{seed_idx:03d}.json"
        with open(split_dir / fname, "w") as f:
            json.dump(payload, f)

    n_train = len(list(train_dir.glob("*.json")))
    n_heldout_total = len(list(heldout_dir.glob("*.json")))
    print(f"\nTraining set: {n_train} campaigns in {train_dir}")
    print(f"Held-out set: {n_heldout_total} campaigns in {heldout_dir}")


if __name__ == "__main__":
    main()
