"""
Confirmatory test: does front-range normalisation help WITHOUT any
domain-tuned beta at all?

Motivation (see .scratch/front-range-robustness-spec/ and the beta-grid
confirmatory run): the full ticket-03 beta sweep {2,5,10,15,25} did not
meet the robustness criterion, and the one beta that did show a
significant advantage (beta=2) happened to equal hint_fixed_ucb's own
fixed beta — raising the concern that any positive result is really
p-hunting over beta rather than evidence the mechanism helps.

This script removes beta as a free/tuned parameter entirely. Instead of
a fixed or swept multiplier, sigma_norm is weighted by the theoretically
motivated GP-UCB confidence schedule (Srinivas et al. 2010, the finite/
discretised-domain form of their Theorem 1):

    beta_t = 2 * log( |pool| * t^2 * pi^2 / (6*delta) )
    weight = sqrt(beta_t)

with delta=0.1 fixed by convention (not tuned), t = number of
observations so far (n_obs), |pool| = candidate pool size at this batch
— all read directly from context, no domain-specific calibration. This
schedule is identical in form for every domain/seed/condition; nothing
here is fit to this synthetic domain's dominance_ratio the way the
original beta=15 pilot value was.

Three conditions, run on the same 8-domain-seed x 20-campaign x
budget=40 regime as the ticket-03 confirmatory run (reuses its domain
config and campaign-tracking machinery directly):
  - hint_fixed_ucb (beta=2, the untuned baseline, unchanged)
  - gen6_child0_beta2 (front-range norm at the same fixed beta=2, for
    reference — this is the condition that showed p=0.003 in the full
    sweep)
  - gen6_child0_gpucb (front-range norm, GP-UCB schedule, NO tuned beta)

Primary test: does gen6_child0_gpucb beat hint_fixed_ucb via a
random-intercept MixedLM (AUC ~ condition + (1|domain_seed)), same as
the ticket-05 primary confirmatory test? Single comparison, so no BH
correction needed. Also reports gen6_child0_beta2 for context (already
known from the beta-grid run, included here only to reconfirm it lands
in the same place on this identical harness) and a cluster bootstrap CI
for the gpucb comparison.
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

_GEN6_CHILD0_GPUCB = '''
def score_pool(context):
    """Front-range-normalised sigma UCB, GP-UCB confidence schedule
    (Srinivas et al. 2010) instead of a fixed/tuned beta. No free
    hyperparameter: delta=0.1 is the schedule's own fixed convention,
    not calibrated to this domain."""
    import math
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    pool_size = max(len(context["pool"]), 2)
    t = max(context["campaign"]["n_obs"], 2)
    delta = 0.1
    beta_t = 2.0 * math.log(pool_size * (t ** 2) * (math.pi ** 2) / (6.0 * delta))
    beta = math.sqrt(max(beta_t, 0.0))
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        scores.append(mu_sum + beta * sigma_norm)
    return scores
'''.strip("\n")


def build_conditions():
    return {
        "hint_fixed_ucb": SEED_HINT_FIXED_UCB_TEMPLATE.format(beta=HINT_BETA),
        "gen6_child0_beta2": _GEN6_CHILD0_TEMPLATE.format(beta=2),
        "gen6_child0_gpucb": _GEN6_CHILD0_GPUCB,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_domains", type=int, default=len(DOMAIN_SEEDS))
    ap.add_argument("--n_campaigns", type=int, default=N_CAMPAIGNS)
    ap.add_argument("--out_path", default=str(HERE / "gpucb_schedule_results.json"))
    ap.add_argument("--raw_out_path", default=str(HERE / "gpucb_schedule_raw.parquet"))
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

    print(f"\n{'='*70}\nPrimary test: gen6_child0_gpucb vs hint_fixed_ucb "
          f"(AUC ~ condition + (1|domain_seed))\n{'='*70}")
    hint_auc = auc[auc["condition"] == "hint_fixed_ucb"]
    for cond_name in ["gen6_child0_beta2", "gen6_child0_gpucb"]:
        sub = pd.concat([auc[auc["condition"] == cond_name], hint_auc]).copy()
        sub["is_gen6"] = (sub["condition"] == cond_name).astype(int)
        md = smf.mixedlm("auc ~ is_gen6", sub, groups=sub["domain_seed"])
        fit = md.fit()
        pval = float(fit.pvalues["is_gen6"])
        effect = float(fit.params["is_gen6"])  # negative = gen6 has LOWER (better) log-diff
        print(f"  {cond_name:>20}: effect={effect:+.4f}  p={pval:.4f}  "
              f"({'favors gen6' if effect < 0 else 'favors hint_fixed_ucb'})")

    # Cluster bootstrap CI for the headline (no-tuning) comparison
    rng = np.random.default_rng(0)
    a = auc[auc["condition"] == "gen6_child0_gpucb"]
    h = hint_auc
    seeds_present = sorted(auc["domain_seed"].unique())
    boot_diffs = []
    for _ in range(2000):
        resample = rng.choice(seeds_present, size=len(seeds_present), replace=True)
        a_mean = np.mean([a[a["domain_seed"] == s]["auc"].mean() for s in resample])
        h_mean = np.mean([h[h["domain_seed"] == s]["auc"].mean() for s in resample])
        boot_diffs.append(a_mean - h_mean)
    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    print(f"\nCluster bootstrap CI, gen6_child0_gpucb - hint_fixed_ucb: "
          f"mean={boot_diffs.mean():+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] "
          f"(negative = gpucb better)")

    with open(args.out_path, "w") as f:
        json.dump({
            "domain_seeds": domain_seeds,
            "bootstrap_ci": {"mean": float(boot_diffs.mean()),
                              "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)},
        }, f, indent=2)
    print(f"\nSaved analysis summary to {args.out_path}")


if __name__ == "__main__":
    main()
