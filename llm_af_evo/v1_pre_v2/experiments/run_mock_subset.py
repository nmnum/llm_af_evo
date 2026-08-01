"""
run_mock_subset.py — llm_af_evo pre-test, step 2: small instrumented mock-LLM
re-run to produce per-step logs for the 2a (single-step counterfactual HV
regret) diagnostic.

Scope, per the decision-complete plan:
  1 protein (mAb_aggregation) x 1 prior (L1) x 2 conditions
  (mo_egbo_novelty, mo_ls_na_egbo) x n_seeds (default 15).

This script does NOT modify any campaign/strategy logic — it only calls the
existing run_mo_campaign / run_ls_na_egbo_campaign entry points (now emitting
picked_x/picked_y and, for mo_egbo_novelty's strategy, the full candidate
pool + GP-predicted objectives + acquisition scores per batch, per the
additive log_extra changes in excipient_campaign_mo.py and
strategy_ls_na_egbo.py) and writes the raw decisions to JSON, one file per
(condition, seed), plus the campaign's initial X_init/Y_init (needed by the
diagnostic to reconstruct the running Pareto front — run_mo_campaign itself
never persists X_init/Y_init, only the final X_obs/Y_obs).

Usage:
    python run_mock_subset.py --n_seeds 15 --out_dir logs
"""

import argparse
import json
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from excipient_oracle_mo import MultiObjectiveExcipientOracle
from excipient_campaign_mo import (
    run_mo_campaign, make_shared_inits, pareto_front_of,
)
from strategy_ls_na_egbo import (
    strategy_mo_egbo_novelty, strategy_mo_ls_na_egbo, run_ls_na_egbo_campaign,
)


PROTEIN = "mAb_aggregation"
PRIOR_LEVEL = "L1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=15)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--n_propose", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--out_dir", default=str(pathlib.Path(__file__).parent / "logs"))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle = MultiObjectiveExcipientOracle(
        protein=PROTEIN, tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10,
        seed=42,
    )
    # Same discrete pool for every seed's random init (mo_egbo_novelty);
    # run_ls_na_egbo_campaign draws its own discrete oracle per seed below,
    # mirroring run_benchmark_resumable.py's run_one_seed exactly, since
    # oracle.make_discrete_oracle() is stateful (draws from oracle.rng) and
    # is not safe to reuse across seeds within a single process run here.
    disc_shared = oracle.make_discrete_oracle(n_samples=500, seed=42)
    shared_inits = make_shared_inits(disc_shared, args.n_seeds, args.n_init, rng_seed=42)

    conditions = {
        "mo_egbo_novelty": (strategy_mo_egbo_novelty, {}, False),
        "mo_ls_na_egbo": (strategy_mo_ls_na_egbo, {}, True),
    }

    for cond_name, (fn, kwargs, is_warmstart) in conditions.items():
        cond_dir = out_dir / cond_name
        cond_dir.mkdir(parents=True, exist_ok=True)
        print(f"{cond_name}...", end="", flush=True)

        for seed_idx in range(args.n_seeds):
            disc_seed = oracle.make_discrete_oracle(n_samples=500, seed=42)

            if is_warmstart:
                from excipient_campaign_mo import MO_PRIORS
                prior_text = MO_PRIORS["agg_L1"]
                X_init, Y_init, forms, warmstart_meta = None, None, None, None
                result = run_ls_na_egbo_campaign(
                    disc_seed, budget=args.budget, strategy_fn=fn,
                    strategy_kwargs=kwargs, batch_size=args.batch_size,
                    seed=seed_idx, n_init=args.n_init, n_propose=args.n_propose,
                    protein=PROTEIN, prior_level=PRIOR_LEVEL,
                    mock_llm=True,
                )
                # run_ls_na_egbo_campaign doesn't return X_init/Y_init
                # directly, but decisions[0]'s n_obs_before_pick == n_init
                # and X_obs[:n_init]/Y_obs[:n_init] IS the warm-start init
                # (run_mo_campaign inside it starts X_obs = X_init.copy()
                # and only appends afterward) — safe to slice here since we
                # know n_init exactly.
                X_init = result["X_obs"][:args.n_init]
                Y_init = result["Y_obs"][:args.n_init]
            else:
                X_init, Y_init = shared_inits[seed_idx]
                result = run_mo_campaign(
                    disc_seed, X_init, Y_init, args.budget, fn, kwargs,
                    batch_size=args.batch_size, seed=seed_idx,
                )

            payload = {
                "seed": seed_idx,
                "condition": cond_name,
                "protein": PROTEIN,
                "prior_level": PRIOR_LEVEL,
                "budget": args.budget,
                "n_init": args.n_init,
                "batch_size": args.batch_size,
                "X_init": np.asarray(X_init).tolist(),
                "Y_init": np.asarray(Y_init).tolist(),
                "hv_trajectory": result["hv_trajectory"],
                "decisions": result["decisions"],
                "final_pf_size": len(pareto_front_of(result["Y_obs"])),
                # disc_seed's pool is redrawn (fresh noise) per seed_idx —
                # oracle.make_discrete_oracle() draws from the stateful,
                # never-reset oracle.rng, so seed_idx N's pool cannot be
                # reproduced later just by calling make_discrete_oracle
                # again with the same seed= kwarg. Dumping it here is what
                # makes the true-oracle-HV diagnostic possible without
                # re-running campaigns: it needs ground-truth y for every
                # pool candidate at every step, not just the picked ones.
                "oracle_X_raw": disc_seed._X_raw.tolist(),
                "oracle_Y_raw": disc_seed._Y_raw.tolist(),
            }
            with open(cond_dir / f"seed_{seed_idx:03d}.json", "w") as f:
                json.dump(payload, f)
            print(".", end="", flush=True)
        print()

    print(f"\nLogs written to {out_dir}")


if __name__ == "__main__":
    main()
