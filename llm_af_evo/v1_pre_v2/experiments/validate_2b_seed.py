"""
validate_2b_seed.py — paired 2b comparison of one AF against EGBO-novelty
on ALL held-out campaigns (not just the 5-campaign sample from
run_2b_diagnostic.py's Step B), with a Wilcoxon signed-rank test.

This exists specifically to check whether a promising Step B result (e.g.
gen5_child7 beating baseline by +11.3% on 5 campaigns) is a real effect or
a 5-campaign-sample artifact — the baseline's own per-campaign HV already
spans 2254-3677 (a 63% range) just from run1's 5-campaign timing pilot, so
a 5-campaign mean is not enough to trust on its own.

Runs the SAME AF and the SAME (unmodified) EGBO-novelty baseline on every
held-out campaign, paired by seed=campaign index, and reports the paired
difference, Wilcoxon p-value, and a per-campaign breakdown so a win driven
by one or two outlier campaigns is visible, not hidden in a mean.

Usage:
    python validate_2b_seed.py --af_id gen5_child7
    python validate_2b_seed.py --seed_name ehvi_approx
    python validate_2b_seed.py --af_path some_other_af.py
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
from af_interface import SEED_PROGRAMS

HERE = pathlib.Path(__file__).parent


def load_af_code(args) -> tuple:
    if args.seed_name:
        if args.seed_name not in SEED_PROGRAMS:
            raise ValueError(f"{args.seed_name!r} not in af_interface.SEED_PROGRAMS "
                              f"({sorted(SEED_PROGRAMS)})")
        return SEED_PROGRAMS[args.seed_name], args.seed_name
    if args.af_path:
        return pathlib.Path(args.af_path).read_text(), args.af_path
    population = json.load(open(args.population_path))
    for p in population:
        if p["id"] == args.af_id:
            return p["code"], args.af_id
    raise ValueError(f"AF id {args.af_id!r} not found in {args.population_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af_id", default="gen5_child7")
    ap.add_argument("--seed_name", default=None,
                     help="Pull code directly from af_interface.SEED_PROGRAMS "
                          "by name (e.g. ehvi_approx) instead of a run's "
                          "final_population.json — takes priority over --af_id.")
    ap.add_argument("--af_path", default=None,
                     help="Use a raw .py file instead of pulling by --af_id "
                          "from --population_path.")
    ap.add_argument("--population_path",
                     default=str(HERE / "evolution_runs" / "run1" / "final_population.json"))
    ap.add_argument("--heldout_dir", default=str(HERE / "training_logs" / "heldout"))
    args = ap.parse_args()

    af_code, af_label = load_af_code(args)
    logs = [json.load(open(f)) for f in sorted(pathlib.Path(args.heldout_dir).glob("*.json"))]
    print(f"Validating {af_label!r} against EGBO-novelty (2b, paired) on "
          f"{len(logs)} held-out campaigns...\n")

    af_hvs, baseline_hvs = [], []
    for i, log in enumerate(logs):
        af_result = run_2b_campaign(af_code, log, seed=i)
        base_result = run_baseline_campaign(log, seed=i)
        af_hvs.append(af_result["final_hv"])
        baseline_hvs.append(base_result["final_hv"])
        diff = af_result["final_hv"] - base_result["final_hv"]
        print(f"  campaign {i:>2}: af={af_result['final_hv']:>8.1f}  "
              f"baseline={base_result['final_hv']:>8.1f}  "
              f"diff={diff:>+8.1f}  {'AF WINS' if diff > 0 else ''}")

    af_hvs, baseline_hvs = np.array(af_hvs), np.array(baseline_hvs)
    diffs = af_hvs - baseline_hvs
    n_wins = int(np.sum(diffs > 0))

    mean_af, mean_base = af_hvs.mean(), baseline_hvs.mean()
    pct_diff = 100 * (mean_af - mean_base) / mean_base

    try:
        stat, pval = wilcoxon(diffs)
    except ValueError:
        pval = 1.0  # all-zero diffs, degenerate

    print(f"\nAF mean final HV:       {mean_af:.1f}")
    print(f"Baseline mean final HV: {mean_base:.1f}")
    print(f"Mean difference:        {mean_af - mean_base:+.1f} ({pct_diff:+.1f}%)")
    print(f"AF wins on {n_wins}/{len(logs)} campaigns")
    print(f"Wilcoxon signed-rank p = {pval:.4f}")

    if pval < 0.05 and mean_af > mean_base:
        print("\n=> SIGNIFICANT WIN: this AF beats EGBO-novelty on held-out "
              "campaigns. This is a positive result on its own — 2b evolution "
              "becomes a 'can we improve on this' follow-up, not the load-"
              "bearing experiment.")
    elif mean_af > mean_base:
        print("\n=> DIRECTIONALLY POSITIVE, NOT SIGNIFICANT: the effect may be "
              "real but small/noisy at this sample size. 2b evolution is "
              "needed to amplify the signal — seed with this AF.")
    else:
        print("\n=> DOES NOT BEAT BASELINE on the full held-out set: the "
              "5-campaign result was likely noise. The 2a proxy is still "
              "confirmed leaky (separately, via the Spearman result), so 2b "
              "evolution is still justified — seed with this AF anyway as "
              "the best direction found so far, and let evolution refine it.")


if __name__ == "__main__":
    main()
