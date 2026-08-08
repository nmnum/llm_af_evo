"""
Confirmatory test of growth-aware modification (1b): fix v2's scale
mismatch (growth_aware_v2's boundary_std was ~0.02-0.08 vs front_range's
~3.4-3.8 — near two orders of magnitude apart, and FLAT across the whole
campaign, not decaying — because it queried GP uncertainty AT an already-
observed front boundary point, where posterior std is near its minimum by
construction of GP interpolation; see diagnose_growth_weight.py's
findings).

This variant sources the growth signal from an UNOBSERVED point instead:
for each objective, the candidate pool already carries
gp_posterior[name]["std"] for every candidate (context["pool"]) — take
whichever candidate has the highest predicted mean for that objective
(the GP's own top pick for extending it, regardless of whether it
actually exceeds front_max — this doesn't reintroduce v1(original)'s
pool_max>front_max gate, which is what made THAT variant structurally
dead) and read that candidate's own std as the growth signal:

    growth_weight_j = 1 + std_of_top_predicted_candidate_j / front_range_j
    score = mu_sum + beta * sum_j [ (std_j / front_range_j) * growth_weight_j ]

Needs no new plumbing beyond what v1/v2 already exposed — pool_mu/
pool_sigma were already being computed and passed through
context["pool"]; this is pure score_pool logic, unlike v2's
pareto_front_boundary_std which needed a new sandbox/full_replay wire.

Five conditions, same 8-seed x 20-campaign x budget=40 regime as the rest
of this spec:
  - hint_fixed_ucb (beta=2, unchanged baseline)
  - gen6_child0_beta2 (original mechanism, for direct reference)
  - gen6_child0_growth_aware_beta2 (v1, pool-extreme, mostly dead)
  - gen6_child0_growth_aware_v2_beta2 (v2, boundary-std, fires but scale-
    mismatched — see module docstring)
  - gen6_child0_growth_aware_v3_beta2 (this variant, pool-top-std)
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
for _p in [HERE.parent / "src", HERE.parent.parent / "shared",
           HERE.parent.parent.parent]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from excipient_campaign_mo import make_shared_inits
from tunable_synthetic_oracle import TunableSyntheticMOOracle

from run_confirmatory_spec import (
    DOMAIN_SEEDS, PLATEAU_SHARPNESS, NOISE_LEVEL, NOISE_MODE, SCALE2,
    N_CAMPAIGNS, BUDGET, HINT_BETA,
    SEED_HINT_FIXED_UCB_TEMPLATE, _GEN6_CHILD0_TEMPLATE,
    compute_true_hv_and_ref, run_one_campaign_tracked,
)
from run_growth_aware import _GEN6_CHILD0_GROWTH_AWARE
from run_growth_aware_v2 import _GEN6_CHILD0_GROWTH_AWARE_V2

_GEN6_CHILD0_GROWTH_AWARE_V3 = '''
def score_pool(context):
    """Front-range-normalised sigma UCB (gen6_child0), with a growth-
    potential weight sourced from the pool candidate with the highest
    predicted mean per objective (modification (1b): fixes v2's scale
    mismatch by reading GP std at an unobserved pool point instead of an
    already-observed front boundary point). Keeps the necessary
    sigma/front_range division for cross-objective scale comparability;
    the growth weight is a separate, unitless factor, not a substitute
    for it. beta={beta}."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    pool = context["pool"]
    beta = {beta}

    growth_weight = {{}}
    for name in names:
        top_std = max(c["gp_posterior"][name]["std"] for c in pool)
        growth_weight[name] = 1.0 + top_std / max(front_range[name], 1e-6)

    scores = []
    for cand in pool:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm_growth = sum(
            (gp[name]["std"] / front_range[name]) * growth_weight[name]
            for name in names)
        scores.append(mu_sum + beta * sigma_norm_growth)
    return scores
'''.strip("\n")


def build_conditions():
    return {
        "hint_fixed_ucb": SEED_HINT_FIXED_UCB_TEMPLATE.format(beta=HINT_BETA),
        "gen6_child0_beta2": _GEN6_CHILD0_TEMPLATE.format(beta=2),
        "gen6_child0_growth_aware_beta2": _GEN6_CHILD0_GROWTH_AWARE.format(beta=2),
        "gen6_child0_growth_aware_v2_beta2": _GEN6_CHILD0_GROWTH_AWARE_V2.format(beta=2),
        "gen6_child0_growth_aware_v3_beta2": _GEN6_CHILD0_GROWTH_AWARE_V3.format(beta=2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_domains", type=int, default=len(DOMAIN_SEEDS))
    ap.add_argument("--n_campaigns", type=int, default=N_CAMPAIGNS)
    ap.add_argument("--out_path", default=str(HERE / "growth_aware_v3_results.json"))
    ap.add_argument("--raw_out_path", default=str(HERE / "growth_aware_v3_raw.parquet"))
    args = ap.parse_args()

    domain_seeds = DOMAIN_SEEDS[:args.n_domains]
    conditions = build_conditions()
    print(f"Domains: {domain_seeds}\nConditions: {list(conditions)}\n"
          f"n_campaigns={args.n_campaigns}, budget={BUDGET}\n")

    rows = []
    for domain_seed in domain_seeds:
        t0 = time.perf_counter()
        oracle = TunableSyntheticMOOracle.build(
            plateau_sharpness=PLATEAU_SHARPNESS, noise_level=NOISE_LEVEL,
            noise_mode=NOISE_MODE, scale2=SCALE2, seed=domain_seed)
        hv_true, ref_point, directions = compute_true_hv_and_ref(oracle)
        inits = make_shared_inits(oracle, args.n_campaigns, 10, rng_seed=domain_seed)

        for cond_name, af_code in conditions.items():
            for camp_i, (X_init, Y_init) in enumerate(inits):
                hv_trace = run_one_campaign_tracked(
                    oracle, X_init, Y_init, hv_true, ref_point, directions,
                    seed=camp_i, af_code=af_code)
                for batch_idx, hv_obs in enumerate(hv_trace):
                    log_hv_diff = float(np.log(max(hv_true - hv_obs, 1e-9)))
                    rows.append({
                        "domain_seed": domain_seed, "condition": cond_name,
                        "campaign": camp_i, "batch": batch_idx,
                        "hv_observed": hv_obs, "hv_true": hv_true,
                        "log_hv_diff": log_hv_diff,
                    })
        print(f"  domain_seed={domain_seed} done in {time.perf_counter()-t0:.1f}s "
              f"(hv_true={hv_true:.4g})")

    df = pd.DataFrame(rows)
    df.to_parquet(args.raw_out_path)
    print(f"\nSaved raw per-batch results to {args.raw_out_path} ({len(df)} rows)")

    auc = (df.groupby(["domain_seed", "condition", "campaign"])["log_hv_diff"]
             .mean().reset_index(name="auc"))

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels not installed — skipping MixedLM stage.")
        return

    print(f"\n{'='*70}\nPrimary test: gen6_child0_growth_aware_v3_beta2 vs hint_fixed_ucb "
          f"(AUC ~ condition + (1|domain_seed))\n{'='*70}")
    hint_auc = auc[auc["condition"] == "hint_fixed_ucb"]
    results = {}
    for cond_name in ["gen6_child0_beta2", "gen6_child0_growth_aware_beta2",
                       "gen6_child0_growth_aware_v2_beta2",
                       "gen6_child0_growth_aware_v3_beta2"]:
        sub = pd.concat([auc[auc["condition"] == cond_name], hint_auc]).copy()
        sub["is_gen6"] = (sub["condition"] == cond_name).astype(int)
        md = smf.mixedlm("auc ~ is_gen6", sub, groups=sub["domain_seed"])
        fit = md.fit()
        pval = float(fit.pvalues["is_gen6"])
        effect = float(fit.params["is_gen6"])  # negative = gen6 has LOWER (better) log-diff
        results[cond_name] = {"effect": effect, "p": pval}
        print(f"  {cond_name:>34}: effect={effect:+.4f}  p={pval:.4f}  "
              f"({'favors gen6' if effect < 0 else 'favors hint_fixed_ucb'})")

    rng = np.random.default_rng(0)
    a_v3 = auc[auc["condition"] == "gen6_child0_growth_aware_v3_beta2"]
    h = hint_auc
    seeds_present = sorted(auc["domain_seed"].unique())
    boot_diffs = []
    for _ in range(2000):
        resample = rng.choice(seeds_present, size=len(seeds_present), replace=True)
        a_mean = np.mean([a_v3[a_v3["domain_seed"] == s]["auc"].mean() for s in resample])
        h_mean = np.mean([h[h["domain_seed"] == s]["auc"].mean() for s in resample])
        boot_diffs.append(a_mean - h_mean)
    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    print(f"\nCluster bootstrap CI, gen6_child0_growth_aware_v3_beta2 - hint_fixed_ucb: "
          f"mean={boot_diffs.mean():+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] "
          f"(negative = growth-aware-v3 better)")

    with open(args.out_path, "w") as f:
        json.dump({
            "domain_seeds": domain_seeds,
            "mixedlm": results,
            "bootstrap_ci": {"mean": float(boot_diffs.mean()),
                              "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)},
        }, f, indent=2)
    print(f"\nSaved analysis summary to {args.out_path}")


if __name__ == "__main__":
    main()
