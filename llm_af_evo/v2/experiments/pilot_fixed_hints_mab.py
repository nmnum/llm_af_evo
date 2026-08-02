"""
Quick pilot: do the 3 newly-fixed hints (noisy_front_hvi, local_penalization,
pareto_membership) show real signal on mAb, the domain where sigma actually
moves rankings? Uses the fixed call_00004/00007/00008 code straight from
run_v2_mAb_gamma001_fixed/af_code_logs (the interrupted run), replayed against
a subset of mAb training campaigns via run_2b_campaign/run_baseline_campaign.

Usage: python pilot_fixed_hints_mab.py [--n-campaigns N] [--n-seeds N] [--out FILE]

Writes full per-campaign/per-seed data plus the summary table to --out
(default pilot_fixed_hints_mab_results.json in this script's directory)
in addition to printing progress, so results can be read back from disk
without needing to capture/paste terminal output.

Multi-seed averaging: each campaign's final_hv (baseline and AF alike) is
averaged over --n-seeds distinct torch seeds before computing the margin,
same rationale as mab_noise_diagnostic.py / N_FITNESS_SEEDS_DEFAULTS in
evolve_af_v2.py -- a single seed's final_hv is dominated by GP-fit/UNSGA3
seed noise on mAb (SE ~0.057 per the noise diagnostic), which is large
enough to manufacture the kind of single-campaign outlier margins seen in
the n=8, single-seed pilot (e.g. the shared spike across all 3 hints on
campaign 2, whose baseline HV was the lowest of the 8 -- a low-denominator
artifact, not independent confirmation).
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
# excipient_oracle_mo.py (and other ls_na_egbo-level oracle/campaign deps)
# live in the real ls_na_egbo repo root, not under llm_af_evo, so they're
# not present in a worktree that only checked out the llm_af_evo subtree.
_LS_NA_EGBO_ROOT = pathlib.Path("/home/nehamungale/ls_na_egbo")
for _p in (
    _ROOT,
    _LS_NA_EGBO_ROOT,
    _LLM_AF_EVO / "shared",
    _LLM_AF_EVO / "v1_pre_v2" / "src",
    _LLM_AF_EVO / "v1_pre_v2" / "experiments",
    _LLM_AF_EVO / "v2" / "src",
    _LLM_AF_EVO / "v2" / "experiments",
):
    sys.path.insert(0, str(_p))

from full_replay import run_2b_campaign, run_baseline_campaign

RUN_DIR = _LLM_AF_EVO / "data" / "evolution_runs" / "run_v2_mAb_gamma001_fixed" / "af_code_logs"
TRAIN_DIR = _LLM_AF_EVO / "data" / "training_logs_mAb_100" / "train"

HINTS = {
    "noisy_front_hvi": RUN_DIR / "call_00004.py",
    "local_penalization": RUN_DIR / "call_00007.py",
    "pareto_membership": RUN_DIR / "call_00008.py",
}


def seed_averaged_hv(run_with_seed, campaign_idx, n_seeds):
    """Average final_hv over n_seeds distinct torch seeds for one campaign.
    run_with_seed(seed) must run the campaign and return the result dict.
    Seeds are offset per campaign (campaign_idx * 1000 + s) so no two
    campaigns share a seed, matching mab_noise_diagnostic.py's decoupling."""
    hvs = []
    for s in range(n_seeds):
        r = run_with_seed(campaign_idx * 1000 + s)
        hvs.append(r["final_hv"])
    return float(np.mean(hvs)), hvs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-campaigns", type=int, default=20)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent /
                                          "pilot_fixed_hints_mab_results.json"))
    args = ap.parse_args()
    out_path = pathlib.Path(args.out)

    results = {"n_campaigns": args.n_campaigns, "n_seeds": args.n_seeds,
               "baseline": {}, "hints": {}}

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    files = sorted(TRAIN_DIR.glob("*.json"))[: args.n_campaigns]
    logs = [json.load(open(f)) for f in files]

    print(f"Running baseline on {len(logs)} mAb campaigns x {args.n_seeds} seeds each...", flush=True)
    baseline_hvs = []
    per_campaign = []
    for i, log in enumerate(logs):
        avg_hv, hvs = seed_averaged_hv(
            lambda seed, log=log: run_baseline_campaign(log, seed=seed, oracle_family="excipient"),
            i, args.n_seeds)
        baseline_hvs.append(avg_hv)
        per_campaign.append({"campaign": i, "avg_hv": avg_hv, "per_seed_hv": hvs})
        print(f"  baseline campaign {i}: avg_hv={avg_hv:.1f}  per_seed={['%.1f' % h for h in hvs]}", flush=True)
    print("baseline done.\n", flush=True)
    results["baseline"] = {"avg_hvs": baseline_hvs, "per_campaign": per_campaign}
    save()

    for name, path in HINTS.items():
        code = path.read_text()
        margins = []
        wins = 0
        per_campaign = []
        for i, log in enumerate(logs):
            avg_hv, hvs = seed_averaged_hv(
                lambda seed, log=log: run_2b_campaign(
                    code, log, seed=seed, sandbox_log_dir=None, oracle_family="excipient"),
                i, args.n_seeds)
            b = baseline_hvs[i]
            margin = (avg_hv - b) / abs(b)
            margins.append(margin)
            if avg_hv > b:
                wins += 1
            per_campaign.append({"campaign": i, "avg_hv": avg_hv, "per_seed_hv": hvs,
                                  "margin": margin})
            print(f"  {name} campaign {i}: avg_hv={avg_hv:.1f} margin={margin * 100:+.2f}%  "
                  f"per_seed={['%.1f' % h for h in hvs]}", flush=True)
            results["hints"][name] = {"per_campaign": per_campaign}
            save()
        margins = np.array(margins)
        try:
            stat, p = wilcoxon(margins)
        except ValueError:
            p = float("nan")
        summary = {"mean_margin": float(margins.mean()), "median_margin": float(np.median(margins)),
                   "wins": wins, "n": len(logs), "p_value": float(p)}
        results["hints"][name]["summary"] = summary
        save()
        print(f"{name:20s} mean_margin={summary['mean_margin'] * 100:+.2f}%  "
              f"median={summary['median_margin'] * 100:+.2f}%  wins={wins}/{len(logs)}  p={p:.4f}")

    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
