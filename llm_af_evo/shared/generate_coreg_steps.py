"""
generate_coreg_steps.py — L1 build, DA-COREG variant: generate the training-
step snapshots the covariance-aware evolution run (§22's Gate 3, "obj
correlation" plan) needs, from real mAb DA-COREG campaigns instead of the
independent-GP mo_egbo_novelty campaigns generate_training_set.py produces.

Why a new script rather than reusing generate_training_set.py: its
campaigns run strategy_mo_egbo_novelty, which always fits independent
per-objective GPs (ModelListGP) — there is no way to get a DA-COREG
posterior, let alone its cross-objective correlation, out of that code
path. evolve_af.py's load_training_steps only ever reads whatever a
campaign log's decisions actually contain (pool_x_norm/pool_pred_mu/
pool_pred_sigma, and now optionally pool_obj_correlation — see
compose_strategies.strategy_ablation_cell) — it has no opinion on which
strategy produced them, so campaigns generated here slot into the exact
same load_training_steps/evaluate_af pipeline unchanged.

Baseline picks in every step here are made by strategy_ablation_cell(
use_da_coreg=True, use_compose=False) — the DA-COREG surrogate + this
project's own qLogNEHVI-score + novelty-weighted selection, NOT plain
qLogNEHVI top-k and NOT strategy_mo_egbo_novelty's independent-GP version.
This is deliberate: evolve_af.py's fitness compares a CANDIDATE af_code's
picks against the baseline's true HV gain at that exact step (see
evaluate_af's docstring) — if the baseline itself still ran on independent
GPs while candidate AFs are scored using the DA-COREG pool_mu/pool_sigma/
obj_correlation, any measured "improvement" would be entangled with the
surrogate swap itself, not attributable to the evolved AF's use of
obj_correlation specifically. Holding the surrogate fixed (DA-COREG on
both sides) isolates that.

Usage:
    python generate_coreg_steps.py --n_seeds_per_level 20 --out_dir training_logs_coreg
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
from compose_strategies import strategy_ablation_cell

PROTEIN = "mAb_aggregation"  # base profile; aggregation_tendency overridden below
AGG_LEVELS = [0.25, 0.50, 0.75, 0.85]
HELDOUT_FRAC = 0.25

STRATEGY_KWARGS = {"use_da_coreg": True, "use_compose": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds_per_level", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--out_dir", default=str(pathlib.Path(__file__).parent /
                                              "training_logs_coreg"))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    train_dir = out_dir / "train"
    heldout_dir = out_dir / "heldout"
    train_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    n_heldout = max(1, int(round(args.n_seeds_per_level * HELDOUT_FRAC)))
    print(f"Per aggregation_tendency level: {args.n_seeds_per_level} seeds "
          f"({args.n_seeds_per_level - n_heldout} train / {n_heldout} heldout)")
    print("Baseline picks: strategy_ablation_cell(use_da_coreg=True, "
          "use_compose=False) — DA-COREG surrogate throughout, same "
          "selection mechanism as strategy_mo_egbo_novelty, so the only "
          "thing evolved AFs will be compared against is the surrogate's "
          "OWN baseline picks under DA-COREG, not an independent-GP "
          "baseline.\n")

    total_fallback, total_batches = 0, 0
    for agg in AGG_LEVELS:
        oracle = MultiObjectiveExcipientOracle(
            protein=PROTEIN, tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10,
            seed=42,
        )
        oracle.protein.aggregation_tendency = agg

        disc_shared = oracle.make_discrete_oracle(n_samples=500, seed=42)
        shared_inits = make_shared_inits(disc_shared, args.n_seeds_per_level,
                                          args.n_init, rng_seed=42)

        print(f"agg={agg}...", end="", flush=True)
        for seed_idx in range(args.n_seeds_per_level):
            disc_seed = oracle.make_discrete_oracle(n_samples=500, seed=42)
            X_init, Y_init = shared_inits[seed_idx]

            result = run_mo_campaign(
                disc_seed, X_init, Y_init, args.budget, strategy_ablation_cell,
                {**STRATEGY_KWARGS, "budget": args.budget},
                batch_size=args.batch_size, seed=seed_idx,
            )

            n_fallback = sum(1 for d in result["decisions"]
                              if d.get("ablation_cell_ok") is False)
            total_fallback += n_fallback
            total_batches += len(result["decisions"])

            payload = {
                "seed": seed_idx,
                "condition": "da_coreg_novelty_select",
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
    if total_batches:
        print(f"DA-COREG fallback (strategy_ablation_cell -> strategy_mo_egbo): "
              f"{total_fallback}/{total_batches} batches — a silent fallback here "
              f"would mean some 'DA-COREG' steps never actually used the "
              f"coregionalized surrogate, contaminating obj_correlation with an "
              f"empty {{}} on those batches. Check this is 0 (or small) before "
              f"trusting a downstream evolution run's use of obj_correlation.")


if __name__ == "__main__":
    main()
