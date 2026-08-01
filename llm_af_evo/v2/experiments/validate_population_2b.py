"""
validate_population_2b.py — the step-3 safety net for the n=8 evolution
strategy: don't trust the evolution's point-estimate winner, validate the
ENTIRE final population on all 20 held-out campaigns under 2b, and pick
the true best from that.

Computes EGBO-novelty's baseline ONCE per campaign (not once per AF) —
validate_2b_seed.py reruns baseline for every AF it checks, which is
correct for a single AF but wasteful for a whole population: N AFs against
the same 20 campaigns needs only 20 baseline runs total, not 20*N. Cost
here is 20 baseline + 20*len(population) AF runs, vs 20*2*len(population)
if reusing validate_2b_seed.py per AF.

For each AF in the population: paired per-campaign HV vs baseline,
Wilcoxon signed-rank p, mean/% difference, win count — the same statistics
validate_2b_seed.py reports for one AF, run for all of them, plus a
ranked summary at the end identifying the true best (by mean HV) and
which AFs (if any) reach significance against the baseline.

Usage:
    python validate_population_2b.py --population_path evolution_runs/run2/final_population.json
    python validate_population_2b.py --population_path ... --oracle excipient \\
        --n_fitness_seeds 3   # noise-mitigated re-validation, see below
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.stats import wilcoxon

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
from full_replay import run_2b_campaign, run_baseline_campaign

HERE = pathlib.Path(__file__).parent

# Same convention as evolve_af_v2.py's FITNESS_SEED_STRIDE/
# N_FITNESS_SEEDS_DEFAULTS — kept as a separate constant here rather than
# importing evolve_af_v2 (this script predates that module's fitness-noise
# fixes and validates populations evolved under EITHER module, so it
# shouldn't import evolve_af_v2-specific machinery), but must match its
# stride value for a given campaign index's seeds to mean the same thing
# in both places if ever compared directly.
FITNESS_SEED_STRIDE = 1000
N_FITNESS_SEEDS_DEFAULTS = {"excipient": 3, "coatings": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population_path",
                     default=str(HERE / "evolution_runs" / "run1" / "final_population.json"))
    ap.add_argument("--heldout_dir", default=None,
                     help="Defaults to training_logs/heldout for --oracle excipient, "
                          "or training_logs_coatings/heldout for --oracle coatings.")
    ap.add_argument("--oracle", choices=["excipient", "coatings"], default="excipient",
                     help="Which oracle family to reconstruct and validate against — "
                          "MUST match whatever --oracle the population being "
                          "validated was evolved under (evolve_af_v2.py's --oracle), "
                          "or this validates the population against the wrong "
                          "oracle entirely. See full_replay.reconstruct_oracle's "
                          "oracle_family parameter.")
    ap.add_argument("--n_fitness_seeds", type=int, default=None,
                     help="Number of (GP-fit, UNSGA3) seed repeats to average per "
                          "held-out campaign, for both baseline and every AF. "
                          "Defaults to N_FITNESS_SEEDS_DEFAULTS[--oracle] (3 for "
                          "excipient/mAb, 1 for coatings) — see "
                          "mab_noise_diagnostic.py and evolve_af_v2.py's "
                          "N_FITNESS_SEEDS_DEFAULTS docstring for why: a single "
                          "seed's paired margin on one mAb campaign had "
                          "seed-to-seed std=0.256, so a single-seed held-out "
                          "validation run's exact win/p-value numbers should be "
                          "treated as noisy point estimates, not precise, unless "
                          "averaged over multiple seeds like this.")
    args = ap.parse_args()

    if args.n_fitness_seeds is None:
        args.n_fitness_seeds = N_FITNESS_SEEDS_DEFAULTS[args.oracle]
        print(f"--n_fitness_seeds not given, using "
              f"N_FITNESS_SEEDS_DEFAULTS[{args.oracle!r}]={args.n_fitness_seeds}")

    heldout_dir = args.heldout_dir or str(
        HERE / ("training_logs_coatings" if args.oracle == "coatings" else "training_logs")
        / "heldout")

    population = json.load(open(args.population_path))
    logs = [json.load(open(f)) for f in sorted(pathlib.Path(heldout_dir).glob("*.json"))]
    print(f"Validating {len(population)} AFs from {args.population_path} against "
          f"EGBO-novelty (2b, paired, oracle={args.oracle}) on {len(logs)} "
          f"held-out campaigns, averaged over {args.n_fitness_seeds} seed(s) "
          f"per campaign...\n")

    print("Baseline (EGBO-novelty, run once, shared across all AFs)...")
    baseline_hvs = []
    for i, log in enumerate(logs):
        seeds = [i + r * FITNESS_SEED_STRIDE for r in range(args.n_fitness_seeds)]
        repeat_hvs = [run_baseline_campaign(log, seed=s, oracle_family=args.oracle)["final_hv"]
                      for s in seeds]
        avg_hv = float(np.mean(repeat_hvs))
        baseline_hvs.append(avg_hv)
        print(f"  campaign {i:>2}: baseline={avg_hv:.1f}"
              + (f"  (mean of {repeat_hvs})" if args.n_fitness_seeds > 1 else ""))
    baseline_hvs = np.array(baseline_hvs)
    print(f"Baseline mean: {baseline_hvs.mean():.1f}\n")

    rows = []
    for p in population:
        af_hvs = []
        for i, log in enumerate(logs):
            seeds = [i + r * FITNESS_SEED_STRIDE for r in range(args.n_fitness_seeds)]
            repeat_hvs = [run_2b_campaign(p["code"], log, seed=s, oracle_family=args.oracle)["final_hv"]
                          for s in seeds]
            af_hvs.append(float(np.mean(repeat_hvs)))
        af_hvs = np.array(af_hvs)
        diffs = af_hvs - baseline_hvs
        n_wins = int(np.sum(diffs > 0))
        pct_diff = 100 * (af_hvs.mean() - baseline_hvs.mean()) / baseline_hvs.mean()
        try:
            _, pval = wilcoxon(diffs)
        except ValueError:
            pval = 1.0

        rows.append({"id": p["id"], "fitness_2a": p.get("fitness"),
                     "mean_hv_2b": float(af_hvs.mean()), "pct_diff": pct_diff,
                     "n_wins": n_wins, "n_campaigns": len(logs), "wilcoxon_p": float(pval)})
        sig = "  <-- SIGNIFICANT WIN" if (pval < 0.05 and af_hvs.mean() > baseline_hvs.mean()) else ""
        print(f"  {p['id']:<16} mean_hv_2b={af_hvs.mean():>9.1f}  "
              f"diff={pct_diff:>+6.1f}%  wins={n_wins}/{len(logs)}  p={pval:.4f}{sig}")

    print(f"\n{'='*70}")
    ranked = sorted(rows, key=lambda r: -r["mean_hv_2b"])
    print("Ranked by mean 2b HV (true best, not 2a fitness):")
    for r in ranked:
        sig = " *SIGNIFICANT*" if (r["wilcoxon_p"] < 0.05 and r["pct_diff"] > 0) else ""
        print(f"  {r['id']:<16} mean_hv_2b={r['mean_hv_2b']:>9.1f}  "
              f"diff={r['pct_diff']:>+6.1f}%  p={r['wilcoxon_p']:.4f}{sig}")

    n_significant = sum(1 for r in rows if r["wilcoxon_p"] < 0.05 and r["pct_diff"] > 0)
    best = ranked[0]
    print(f"\nBest by mean 2b HV: {best['id']} ({best['pct_diff']:+.1f}%, p={best['wilcoxon_p']:.4f})")
    if n_significant > 0:
        print(f"=> {n_significant}/{len(population)} AFs beat EGBO-novelty significantly "
              f"(p<0.05). Positive result — scale up to confirm.")
    elif best["pct_diff"] > 0:
        # n=len(logs), not a hardcoded "20" — the original excipient pipeline
        # used 20 held-out campaigns, but a smaller held-out set (e.g.
        # coatings' default 5) has much weaker Wilcoxon power: at n=5 the
        # minimum achievable two-sided p is 2*(1/2)^5=0.0625, so "not
        # significant" here can mean "this test literally cannot detect any
        # effect at this sample size" rather than "no real effect" — worth
        # saying explicitly rather than implying a fixed, always-adequate n.
        min_p_at_n = 2 * (0.5 ** len(logs)) if logs else float("nan")
        print(f"=> Best AF is directionally positive but not significant at "
              f"n={len(logs)} (Wilcoxon's own floor at this n is p>={min_p_at_n:.4f} "
              f"even for a perfect win streak). Marginal — consider whether to "
              f"scale up held-out campaigns or treat as null.")
    else:
        print("=> No AF in this population beats baseline, even directionally. "
              "This evolution run did not find a breakthrough — consider the "
              "prompt/two-model-pipeline follow-ups, or treat as a negative result.")


if __name__ == "__main__":
    main()
