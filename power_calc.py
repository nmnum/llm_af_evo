"""
power_calc.py — Required-n power calculation for the primary claim
(mo_ls_na_egbo vs mo_egbo_novelty, final HV), using observed Phase 1
mock-LLM effect sizes.

Primary claim (Success Criterion 1, scoped per the grilling session):
  mo_ls_na_egbo significantly beats mo_egbo_novelty on final HV,
  p<0.05 after Holm correction across protein x prior cells, using a
  paired (same-seed) test.

Phase 1's mo_ls_na_egbo and mo_egbo_novelty do NOT share initial points
(LLM warm-start vs shared LHS init), so the seed-paired Wilcoxon test
run in run_benchmark_resumable.py is only approximately paired. This
script uses the observed per-seed HV distributions to estimate Cohen's
d and back out the seed count an unpaired (Mann-Whitney-equivalent)
comparison would need for 80% power at a Holm-style corrected alpha.

Usage:
    python power_calc.py --raw_csv results/benchmark_phase1/phase1_raw_results.csv \
        --baseline mo_egbo_novelty --condition mo_ls_na_egbo \
        --n_comparisons 6 --power 0.8
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD standardized mean difference."""
    n1, n2 = len(a), len(b)
    s_pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if s_pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / s_pooled)


def required_n_two_sample(d: float, alpha: float, power: float) -> int:
    """
    Required n PER GROUP for a two-sample t-test to detect effect size d
    at significance alpha (two-sided) with the given power.
    Solved by search using the noncentral t distribution (exact for
    a two-sample t-test), since statsmodels isn't available here.
    """
    if abs(d) < 1e-9:
        return 10**9  # cannot detect a null effect at any finite n

    for n in range(2, 100000):
        df = 2 * n - 2
        ncp = d * np.sqrt(n / 2)
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        # power = P(|T| > t_crit) under noncentral t with ncp
        achieved_power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        if achieved_power >= power:
            return n
    return -1  # not found under search cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", default="results/benchmark_phase1/phase1_raw_results.csv")
    ap.add_argument("--baseline", default="mo_egbo_novelty")
    ap.add_argument("--condition", default="mo_ls_na_egbo")
    ap.add_argument("--metric", default="final_hv")
    ap.add_argument("--n_comparisons", type=int, default=6,
                     help="Number of hypotheses in the Holm family (conditions "
                          "compared against the baseline); used for a Bonferroni-"
                          "style worst-case alpha, since Holm's effective alpha "
                          "depends on the realized p-value ordering.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    args = ap.parse_args()

    df = pd.read_csv(args.raw_csv)
    alpha_corrected = args.alpha / args.n_comparisons

    print(f"Raw results: {args.raw_csv}")
    print(f"Comparison: {args.condition} vs {args.baseline} on {args.metric}")
    print(f"Target power: {args.power}, family-wise alpha: {args.alpha}, "
          f"per-comparison alpha (Bonferroni worst case over "
          f"{args.n_comparisons} comparisons): {alpha_corrected:.5f}\n")

    print(f"{'protein':<18}{'prior':<8}{'n (obs)':>9}{'d':>8}{'req n/grp':>12}"
          f"{'req n/grp @0.05':>18}")

    for (protein, prior), g in df.groupby(["protein", "prior_level"]):
        a = g.loc[g["condition"] == args.condition, args.metric].dropna().to_numpy()
        b = g.loc[g["condition"] == args.baseline, args.metric].dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        d = cohens_d(a, b)
        n_req_corrected = required_n_two_sample(d, alpha_corrected, args.power)
        n_req_uncorrected = required_n_two_sample(d, args.alpha, args.power)
        print(f"{protein:<18}{prior:<8}{len(a):>9}{d:>8.3f}{n_req_corrected:>12}"
              f"{n_req_uncorrected:>18}")

    print("\nNote: these are per-cell two-sample-equivalent estimates from mock-LLM "
          "effect sizes. Real-LLM effect sizes may differ (larger if the rewritten "
          "prompt carries genuine signal, smaller if LLM sampling noise adds variance "
          "per Q8). Treat this as a lower bound on the seed count needed, not a guarantee.")


if __name__ == "__main__":
    main()
