"""
Do the three sign-fixed UCB-diversity "merge" hints -- call_00016, call_00021,
call_00030 from run_v2_mAb_gamma001_fixed's later generations (post-gen-4/7/8,
never fitness-scored because that run was interrupted at gen 11/20) -- show
real signal on mAb once their directional-inversion bugs are corrected?

Each was independently found to invert a distance/proximity sign the same way
call_00007 (local_penalization) and call_00008 (pareto_membership) already
did in Part 4's pilot: 00016 and 00030 rewarded proximity to already-observed
points instead of penalizing it, 00021 counted a sample as "improving" the
Pareto front exactly when it did NOT dominate any existing front point
(backwards). Fixed versions live in fixed_merge_hints/; this script replays
them against mAb the same way pilot_fixed_hints_mab.py replayed the three
originally-broken hints -- n=20 held-out campaigns, 3-seed-averaged per
campaign, using data/training_logs_mAb_100/heldout (genuinely disjoint from
the run_v2_mAb_gamma001_fixed training data, not just a training-set subset).

Usage: python pilot_merge_hints_mab.py [--n-campaigns N] [--n-seeds N] [--out FILE]
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

FIXED_DIR = _LLM_AF_EVO / "v2" / "experiments" / "fixed_merge_hints"
HELDOUT_DIR = _LLM_AF_EVO / "data" / "training_logs_mAb_100" / "heldout"

HINTS = {
    "call_00016_fixed": FIXED_DIR / "call_00016_fixed.py",
    "call_00021_fixed": FIXED_DIR / "call_00021_fixed.py",
    "call_00030_fixed": FIXED_DIR / "call_00030_fixed.py",
}


def seed_averaged_hv(run_with_seed, campaign_idx, n_seeds):
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
                                          "pilot_merge_hints_mab_results.json"))
    args = ap.parse_args()
    out_path = pathlib.Path(args.out)

    results = {"n_campaigns": args.n_campaigns, "n_seeds": args.n_seeds,
               "baseline": {}, "hints": {}}

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    files = sorted(HELDOUT_DIR.glob("*.json"))[: args.n_campaigns]
    logs = [json.load(open(f)) for f in files]

    print(f"Running baseline on {len(logs)} held-out mAb campaigns x {args.n_seeds} seeds each...", flush=True)
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
