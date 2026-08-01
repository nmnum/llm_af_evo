"""
run_generation_pilot.py — candidate-GENERATION gate (Gate 1: geometry-only,
see gen_interface.py's module docstring for the EDGAR-analog rationale).
Tests whether swapping L's fixed qLogNEHVI+UNSGA3 candidate pool for a
geometry-only sandboxed proposer — no GP/acquisition access at all — can
close the gap to baseline, isolating candidate generation the same way
run_composition_pilot.py isolated selection and the original L work
isolated scoring.

Conditions (fixed scoring=trust_only, fixed selection=top-k throughout —
only the candidate-generation step varies):
  mo_egbo_novelty              — baseline: qLogNEHVI scoring + novelty selection (unmodified)
  trust_only_topk               — control: EXISTING qLogNEHVI+UNSGA3 generation + trust_only
                                   scoring + top-k selection (this is the fixed-generation
                                   reference cell — identical to the composition pilot's
                                   trust_only_topk condition, reused here as the control)
  trust_only_gen_lhs            — LHS-in-unexplored-region generation + trust_only + top-k
  trust_only_gen_perturb        — perturb-around-Pareto-front-extremes generation + trust_only + top-k
  trust_only_gen_hybrid         — half-exploit/half-explore generation + trust_only + top-k

Decision gate (pre-committed, matching the composition pilot's discipline):
if any gen_* condition consistently (across 3 replicates) beats
trust_only_topk AND matches/beats mo_egbo_novelty, that's evidence
candidate generation — not scoring, not selection — is where value could be
recovered, motivating either LLM evolution of generation or a GP-informed
Gate 2 proposer. A null here (matching the scoring and selection nulls
already found) would mean none of the three decomposed pipeline stages
(generate / score / select) individually explains or fixes L's tie —
strengthening the "joint MC-integrated batch treatment is the binding
constraint" conclusion.

Usage:
    python run_generation_pilot.py --n_replicates 3 --n_campaigns 20
    python run_generation_pilot.py --n_replicates 1 --n_campaigns 3   # timing smoke test
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
from scipy.stats import wilcoxon

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import torch
from excipient_oracle_mo import MultiObjectiveExcipientOracle
from excipient_campaign_mo import run_mo_campaign, make_shared_inits
from strategy_ls_na_egbo import strategy_mo_egbo_novelty
from full_replay import strategy_evolved_af, strategy_evolved_generation
from af_interface import SEED_PROGRAMS
from gen_interface import GEN_SEED_PROGRAMS

HERE = pathlib.Path(__file__).parent
PROTEIN = "mAb_aggregation"

CONDITIONS = {
    "mo_egbo_novelty": (strategy_mo_egbo_novelty, {}),
    "trust_only_topk": (strategy_evolved_af, {"af_code": SEED_PROGRAMS["trust_only"]}),
    "trust_only_gen_lhs": (strategy_evolved_generation,
                            {"gen_code": GEN_SEED_PROGRAMS["gen_lhs_unexplored"],
                             "af_code": SEED_PROGRAMS["trust_only"]}),
    "trust_only_gen_perturb": (strategy_evolved_generation,
                                {"gen_code": GEN_SEED_PROGRAMS["gen_perturb_front_extremes"],
                                 "af_code": SEED_PROGRAMS["trust_only"]}),
    "trust_only_gen_hybrid": (strategy_evolved_generation,
                               {"gen_code": GEN_SEED_PROGRAMS["gen_half_exploit_half_explore"],
                                "af_code": SEED_PROGRAMS["trust_only"]}),
}


def run_one(oracle, X_init, Y_init, budget, batch_size, seed, fn, kwargs):
    torch.manual_seed(seed)
    kwargs = dict(kwargs)
    if fn in (strategy_evolved_af, strategy_evolved_generation):
        kwargs["budget"] = budget
    result = run_mo_campaign(oracle, X_init, Y_init, budget, fn, kwargs,
                              batch_size=batch_size, seed=seed)
    return result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")


def run_one_replicate(oracle, n_campaigns, budget, n_init, batch_size, replicate_idx,
                       base_seed):
    seed_offset = base_seed + replicate_idx * 1000
    inits = make_shared_inits(oracle, n_campaigns, n_init, rng_seed=seed_offset)

    results = {cond: [] for cond in CONDITIONS}
    for cond_name, (fn, kwargs) in CONDITIONS.items():
        for i, (X_init, Y_init) in enumerate(inits):
            final_hv = run_one(oracle, X_init, Y_init, budget, batch_size,
                                seed_offset + i, fn, kwargs)
            results[cond_name].append(final_hv)
    return results


def summarize(results, n_campaigns):
    baseline = np.array(results["mo_egbo_novelty"])
    control = np.array(results["trust_only_topk"])
    out = {}
    for cond_name in [c for c in CONDITIONS if c != "mo_egbo_novelty"]:
        af_hv = np.array(results[cond_name])
        diffs = af_hv - baseline
        pct = 100 * (af_hv.mean() - baseline.mean()) / baseline.mean()
        n_wins = int(np.sum(diffs > 0))
        try:
            _, pval = wilcoxon(diffs)
        except ValueError:
            pval = 1.0
        out[cond_name] = {"pct_diff": float(pct), "n_wins": n_wins,
                           "n_campaigns": n_campaigns, "p": float(pval)}
    # Gate question: does each generation variant beat the fixed-generation
    # control (trust_only_topk), not just baseline?
    for gen_name in ["trust_only_gen_lhs", "trust_only_gen_perturb", "trust_only_gen_hybrid"]:
        gen_hv = np.array(results[gen_name])
        diffs = gen_hv - control
        pct = 100 * (gen_hv.mean() - control.mean()) / control.mean()
        n_wins = int(np.sum(diffs > 0))
        try:
            _, pval = wilcoxon(diffs)
        except ValueError:
            pval = 1.0
        out[f"{gen_name}_vs_control"] = {"pct_diff": float(pct), "n_wins": n_wins,
                                          "n_campaigns": n_campaigns, "p": float(pval)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_replicates", type=int, default=3)
    ap.add_argument("--n_campaigns", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--out_path", default=str(HERE / "generation_pilot_results.json"))
    args = ap.parse_args()

    oracle_full = MultiObjectiveExcipientOracle(
        protein=PROTEIN, tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10, seed=42)
    oracle = oracle_full.make_discrete_oracle(n_samples=500, seed=42)
    print(f"Oracle: {len(oracle)} pool points, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")
    print(f"Conditions: {list(CONDITIONS)}")
    print(f"Running {args.n_replicates} independent replicates of "
          f"{args.n_campaigns} campaigns x {len(CONDITIONS)} conditions...\n")

    if args.n_campaigns < 20:
        print(f"NOTE: n={args.n_campaigns} — Wilcoxon has essentially no power this "
              f"small. Use this run for a timing/smoke check only.\n")

    all_replicates = []
    for r in range(args.n_replicates):
        t0 = time.perf_counter()
        results = run_one_replicate(oracle, args.n_campaigns, args.budget,
                                     args.n_init, args.batch_size, r, args.base_seed)
        summary = summarize(results, args.n_campaigns)
        elapsed = time.perf_counter() - t0
        print(f"Replicate {r} ({elapsed:.0f}s):")
        for cond_name, s in summary.items():
            print(f"  {cond_name:<32} diff={s['pct_diff']:+.1f}%  "
                  f"wins={s['n_wins']}/{s['n_campaigns']}  p={s['p']:.4f}")
        all_replicates.append({"replicate": r, "per_campaign_final_hv": results,
                                "summary": summary})

    print(f"\n{'='*70}")
    print("ACROSS-REPLICATE PATTERN:")
    all_keys = list(all_replicates[0]["summary"].keys())
    for cond_name in all_keys:
        pcts = [rep["summary"][cond_name]["pct_diff"] for rep in all_replicates]
        wins = [rep["summary"][cond_name]["n_wins"] for rep in all_replicates]
        ps = [rep["summary"][cond_name]["p"] for rep in all_replicates]
        n_sig_positive = sum(1 for i in range(len(pcts)) if ps[i] < 0.05 and pcts[i] > 0)
        n_sig_negative = sum(1 for i in range(len(pcts)) if ps[i] < 0.05 and pcts[i] < 0)
        n_positive = sum(1 for p in pcts if p > 0)
        print(f"\n{cond_name}:")
        print(f"  % diff per replicate: {[f'{p:+.1f}%' for p in pcts]}")
        print(f"  wins per replicate:   {wins}")
        print(f"  p per replicate:      {[f'{p:.3f}' for p in ps]}")
        print(f"  mean % diff across replicates: {np.mean(pcts):+.1f}% "
              f"(std {np.std(pcts):.1f})")
        print(f"  directionally positive in {n_positive}/{len(pcts)} replicates")
        print(f"  significant (p<0.05) POSITIVE in {n_sig_positive}/{len(pcts)} replicates")
        print(f"  significant (p<0.05) NEGATIVE in {n_sig_negative}/{len(pcts)} replicates")
        if n_positive == len(pcts) or n_positive == 0:
            print(f"  => CONSISTENT direction across all replicates.")
        else:
            print(f"  => INCONSISTENT direction across replicates.")

    print(f"\n{'='*70}")
    print("GATE READ: does each generation variant beat the fixed-generation "
          "control (trust_only_topk), AND match/beat baseline?")
    for gen_name in ["trust_only_gen_lhs", "trust_only_gen_perturb", "trust_only_gen_hybrid"]:
        gate_key = f"{gen_name}_vs_control"
        gate_pcts = [rep["summary"][gate_key]["pct_diff"] for rep in all_replicates]
        vs_base_pcts = [rep["summary"][gen_name]["pct_diff"] for rep in all_replicates]
        print(f"\n{gen_name}: vs control = {[f'{p:+.1f}%' for p in gate_pcts]}  "
              f"| vs baseline = {[f'{p:+.1f}%' for p in vs_base_pcts]}")

    with open(args.out_path, "w") as f:
        json.dump(all_replicates, f, indent=2)
    print(f"\nSaved full replicate data to {args.out_path}")


if __name__ == "__main__":
    main()
