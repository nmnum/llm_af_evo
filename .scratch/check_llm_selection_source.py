"""
check_llm_selection_source.py — diagnostic check: in mo_llm_candidate_gen's
"fair" pooled comparison (LLM candidates + EGBO evolutionary/gradient
candidates, scored together by qLogNEHVI, top-batch_size wins regardless of
source), how often do LLM-sourced candidates actually WIN selection, and how
does their acquisition value compare to EGBO-sourced candidates in the same
pool?

This does not change any published result — it only reads the new
diagnostic fields (n_llm_selected, n_egbo_selected, llm_acq_mean/max,
egbo_acq_mean/max) added to strategy_mo_llm_candidate_gen's per-batch
return, and reruns campaigns under the exact Phase 2 recipe (protein,
prior, budget, n_init, batch_size, n_llm_candidates) to inspect them.

Ad-hoc, not part of the published pipeline. Scratch only.
"""
import sys
import time
import pathlib
import json
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from excipient_oracle_mo import MultiObjectiveExcipientOracle
from excipient_campaign_mo import (
    run_mo_campaign, make_shared_inits, MO_PRIORS,
)
from strategy_llm_candidate_gen import strategy_mo_llm_candidate_gen

PROTEIN = "mAb_aggregation"
PRIOR_LEVEL = "L1"
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MOCK_LLM = "--mock" in sys.argv
_model_args = [a for a in sys.argv if a.startswith("--model=")]
MODEL = _model_args[0].split("=", 1)[1] if _model_args else "qwen3:32b"
BUDGET = 40
N_INIT = 10
BATCH_SIZE = 5
N_LLM_CANDIDATES = 25  # matches Phase 2's mo_llm_candidate_gen kwargs


def main():
    oracle = MultiObjectiveExcipientOracle(
        protein=PROTEIN, tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10,
        seed=42,
    )
    disc = oracle.make_discrete_oracle(n_samples=500, seed=42)
    shared_inits = make_shared_inits(disc, N_SEEDS, N_INIT, rng_seed=42)

    prior_key = "agg_L1" if "aggregation" in PROTEIN else "oxid_L1"
    prior_text = MO_PRIORS.get(prior_key, MO_PRIORS["blank"])

    kwargs = {
        "prior_text": prior_text, "mock_llm": MOCK_LLM,
        "model": MODEL, "n_llm_candidates": N_LLM_CANDIDATES,
    }

    all_batches = []
    t_start = time.time()
    for seed_idx in range(N_SEEDS):
        disc_seed = oracle.make_discrete_oracle(n_samples=500, seed=42)
        X_init, Y_init = shared_inits[seed_idx]
        t0 = time.time()
        result = run_mo_campaign(
            disc_seed, X_init, Y_init, BUDGET,
            strategy_mo_llm_candidate_gen, kwargs,
            batch_size=BATCH_SIZE, seed=seed_idx,
        )
        elapsed = time.time() - t0
        print(f"seed {seed_idx}: {elapsed:.0f}s, final_hv="
              f"{result['hv_trajectory'][-1] if result['hv_trajectory'] else float('nan'):.1f}")

        for b_i, d in enumerate(result["decisions"]):
            row = {
                "seed": seed_idx, "batch": b_i,
                "n_llm_candidates": d.get("n_llm_candidates"),
                "n_egbo_candidates": d.get("n_egbo_candidates"),
                "n_llm_selected": d.get("n_llm_selected"),
                "n_egbo_selected": d.get("n_egbo_selected"),
                "llm_acq_mean": d.get("llm_acq_mean"),
                "llm_acq_max": d.get("llm_acq_max"),
                "egbo_acq_mean": d.get("egbo_acq_mean"),
                "egbo_acq_max": d.get("egbo_acq_max"),
                "llm_fallback_used": d.get("llm_fallback_used"),
            }
            all_batches.append(row)

    total_elapsed = time.time() - t_start
    print(f"\nTotal: {total_elapsed:.0f}s for {N_SEEDS} seeds "
          f"({len(all_batches)} batches)\n")

    out_path = pathlib.Path(__file__).parent / "llm_selection_source_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(all_batches, f, indent=2)
    print(f"Saved raw per-batch rows to {out_path}\n")

    # ── Aggregate ──
    n_llm_sel = sum(r["n_llm_selected"] or 0 for r in all_batches)
    n_egbo_sel = sum(r["n_egbo_selected"] or 0 for r in all_batches)
    n_total_sel = n_llm_sel + n_egbo_sel
    n_llm_offered = sum(r["n_llm_candidates"] or 0 for r in all_batches)
    n_egbo_offered = sum(r["n_egbo_candidates"] or 0 for r in all_batches)
    fallback_batches = sum(1 for r in all_batches if r["llm_fallback_used"])

    llm_acq_means = [r["llm_acq_mean"] for r in all_batches
                      if r["llm_acq_mean"] is not None and not np.isnan(r["llm_acq_mean"])]
    egbo_acq_means = [r["egbo_acq_mean"] for r in all_batches
                       if r["egbo_acq_mean"] is not None and not np.isnan(r["egbo_acq_mean"])]
    llm_acq_maxes = [r["llm_acq_max"] for r in all_batches
                      if r["llm_acq_max"] is not None and not np.isnan(r["llm_acq_max"])]
    egbo_acq_maxes = [r["egbo_acq_max"] for r in all_batches
                       if r["egbo_acq_max"] is not None and not np.isnan(r["egbo_acq_max"])]

    print("=" * 60)
    print("SELECTION-BY-SOURCE DIAGNOSTIC")
    print("=" * 60)
    print(f"Batches analysed: {len(all_batches)}  "
          f"(LLM fallback used in {fallback_batches} of them)")
    print(f"\nPool composition (offered): "
          f"LLM {n_llm_offered}, EGBO {n_egbo_offered} "
          f"({100*n_llm_offered/(n_llm_offered+n_egbo_offered+1e-9):.1f}% LLM)")
    print(f"Winners (selected):         "
          f"LLM {n_llm_sel}, EGBO {n_egbo_sel} "
          f"({100*n_llm_sel/(n_total_sel+1e-9):.1f}% LLM)")
    print(f"\nMean per-batch acquisition value (candidate pool):")
    print(f"  LLM-sourced:  mean of batch-means = "
          f"{np.mean(llm_acq_means) if llm_acq_means else float('nan'):.4f}  "
          f"(n_batches={len(llm_acq_means)})")
    print(f"  EGBO-sourced: mean of batch-means = "
          f"{np.mean(egbo_acq_means) if egbo_acq_means else float('nan'):.4f}  "
          f"(n_batches={len(egbo_acq_means)})")
    print(f"\nMax per-batch acquisition value (best candidate offered by each source):")
    print(f"  LLM-sourced:  mean of batch-maxes = "
          f"{np.mean(llm_acq_maxes) if llm_acq_maxes else float('nan'):.4f}")
    print(f"  EGBO-sourced: mean of batch-maxes = "
          f"{np.mean(egbo_acq_maxes) if egbo_acq_maxes else float('nan'):.4f}")

    n_batches_llm_never_selected = sum(
        1 for r in all_batches
        if (r["n_llm_candidates"] or 0) > 0 and (r["n_llm_selected"] or 0) == 0
    )
    print(f"\nBatches where LLM offered >=1 candidate but WON ZERO selections: "
          f"{n_batches_llm_never_selected} / "
          f"{sum(1 for r in all_batches if (r['n_llm_candidates'] or 0) > 0)}")


if __name__ == "__main__":
    main()
