"""
generate_training_set.py — llm_af_evo L1 build, step 1: generate the fixed
training set of mock-LLM mo_egbo_novelty campaigns that L1's AF evolution
trains and validates against.

Sweeps aggregation_tendency in {0.25, 0.50, 0.75, 0.85} (also feeds approach
J, the oxidation-asymmetry investigation, from the same logs) so the trained
AF is not overfit to one protein's dynamics. mo_egbo_novelty ONLY — the
validation gate compares evolved AFs against this baseline specifically, so
mo_ls_na_egbo (warm-start) campaigns aren't needed here.

Reuses run_mo_campaign/strategy_mo_egbo_novelty unmodified, same as
run_mock_subset.py, and the same oracle_X_raw/oracle_Y_raw-dumping pattern
(the discrete pool is redrawn with fresh noise per seed, so it must be
captured per seed for later true-y lookups where needed).

A held-out split is written directly (train/ and heldout/ subdirectories)
so evolve_af.py and validate_l1.py never have to coordinate which seeds
belong to which set out-of-band.

Usage:
    python generate_training_set.py --n_seeds_per_level 20 --out_dir training_logs
    # 4 aggregation_tendency levels x 20 seeds = 80 campaigns total
"""

import argparse
import json
import pathlib
import sys

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

from excipient_oracle_mo import MultiObjectiveExcipientOracle
from excipient_campaign_mo import run_mo_campaign, make_shared_inits, pareto_front_of
from strategy_ls_na_egbo import strategy_mo_egbo_novelty

PROTEIN = "mAb_aggregation"  # base profile; aggregation_tendency overridden below
AGG_LEVELS = [0.25, 0.50, 0.75, 0.85]
HELDOUT_FRAC = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds_per_level", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--out_dir", default=str(pathlib.Path(__file__).parent / "training_logs"))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    train_dir = out_dir / "train"
    heldout_dir = out_dir / "heldout"
    train_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    n_heldout = max(1, int(round(args.n_seeds_per_level * HELDOUT_FRAC)))
    print(f"Per aggregation_tendency level: {args.n_seeds_per_level} seeds "
          f"({args.n_seeds_per_level - n_heldout} train / {n_heldout} heldout)")

    for agg in AGG_LEVELS:
        oracle = MultiObjectiveExcipientOracle(
            protein=PROTEIN, tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10,
            seed=42,
        )
        # Override aggregation_tendency on the (mutable dataclass) protein
        # profile AFTER construction, BEFORE any discrete-oracle draw — every
        # downstream _score_batch call reads self.protein.aggregation_tendency
        # dynamically, so this propagates correctly to every seed's pool.
        oracle.protein.aggregation_tendency = agg

        disc_shared = oracle.make_discrete_oracle(n_samples=500, seed=42)
        shared_inits = make_shared_inits(disc_shared, args.n_seeds_per_level,
                                          args.n_init, rng_seed=42)

        print(f"agg={agg}...", end="", flush=True)
        for seed_idx in range(args.n_seeds_per_level):
            disc_seed = oracle.make_discrete_oracle(n_samples=500, seed=42)
            X_init, Y_init = shared_inits[seed_idx]

            result = run_mo_campaign(
                disc_seed, X_init, Y_init, args.budget, strategy_mo_egbo_novelty, {},
                batch_size=args.batch_size, seed=seed_idx,
            )

            payload = {
                "seed": seed_idx,
                "condition": "mo_egbo_novelty",
                "protein": PROTEIN,
                "aggregation_tendency": agg,
                "budget": args.budget,
                "n_init": args.n_init,
                "batch_size": args.batch_size,
                "X_init": np.asarray(X_init).tolist(),
                "Y_init": np.asarray(Y_init).tolist(),
                "hv_trajectory": result["hv_trajectory"],
                "decisions": result["decisions"],
                "final_pf_size": len(pareto_front_of(result["Y_obs"])),
                "oracle_X_raw": disc_seed._X_raw.tolist(),
                "oracle_Y_raw": disc_seed._Y_raw.tolist(),
            }

            split_dir = heldout_dir if seed_idx < n_heldout else train_dir
            fname = f"agg{agg:.2f}_seed{seed_idx:03d}.json"
            with open(split_dir / fname, "w") as f:
                json.dump(payload, f)
            print(".", end="", flush=True)
        print()

    n_train = len(list(train_dir.glob("*.json")))
    n_heldout_total = len(list(heldout_dir.glob("*.json")))
    print(f"\nTraining set: {n_train} campaigns in {train_dir}")
    print(f"Held-out set: {n_heldout_total} campaigns in {heldout_dir}")


if __name__ == "__main__":
    main()
