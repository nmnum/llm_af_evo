"""
run_confirmatory_spec.py — executes the confirmatory experiment spec locked
down by the wayfinder map at .scratch/front-range-robustness-spec/ (all 7
tickets resolved; see map.md and issues/01-07 for the full rationale behind
every design choice below). Nothing here should require a design judgment
call — every knob traces to a specific resolved ticket.

Hypothesis under test: front-range-normalised sigma UCB (gen6_child0)
accelerates early-campaign HV convergence vs. fixed-beta raw-sigma UCB, an
advantage that shrinks toward budget-independence as budget grows.

Design (ticket 02 domain-seed-replication, ticket 03 beta-sweep-grid):
  - 8 independent domain seeds (TunableSyntheticMOOracle.build(seed=...)),
    domain params FIXED across all of them (plateau_sharpness=5.0,
    noise_level=0.08, scale2=3.0 — sweep_tunable_domain.py's chosen working
    point) — isolates "replicates across independent pool draws" from a
    param sweep.
  - 20 campaigns per domain seed (n_init=10, batch_size=5, budget=40).
  - Conditions: hint_fixed_ucb (beta=2.0, FIXED — not swept, see ticket 03),
    gen6_child0 swept over beta in {2, 5, 10, 15, 25}, phase_decaying_ucb
    (ticket 04's mechanism-isolation baseline, no beta).
  - Total: 8 domains x 20 campaigns x 7 conditions x ~6 batches x ~1s/batch
    ~= 2-2.5 hours. Background-runnable.

Endpoints (ticket 01 primary-endpoint-metric):
  - PRIMARY: AUC of log hypervolume difference vs. the TRUE Pareto front
    (log(HV_true - HV_observed(batch)), averaged over batches per
    campaign) — matches BoTorch's own MOBO benchmarking convention.
    HV_true is computed once per domain seed from the oracle's noiseless
    _Y_true pool (available only because this is a synthetic domain — see
    ticket 06, real-domain tie-back ruled out for exactly this reason among
    others).
  - SECONDARY: batches-to-90%-of-true-optimum-HV (first batch where
    HV_observed >= 0.9 * HV_true; censored at n_batches if never reached).
  - TERTIARY: final HV_observed at budget=40 (carries the existing null
    result — see the pilot note below).

Statistics (ticket 05 statistical-correction-method):
  - Primary confirmatory test: AUC ~ condition + (1 | domain_seed) MixedLM
    (random-intercept), one fit per beta value in the grid, comparing
    gen6_child0(beta) against hint_fixed_ucb(beta=2.0, fixed).
  - Beta-grid robustness (ticket 03): Benjamini-Hochberg correction across
    the 5 beta-level MixedLM p-values. Robust if same-sign effect in >=4/5
    betas AND BH-significant in >=3/5.
  - CIs: cluster bootstrap resampling domain-seeds with replacement (all 20
    campaigns per resampled domain-seed move together) — NOT a flat bootstrap
    over pooled campaigns, which would understate variance given campaigns
    are nested within domain-seed pools.
  - Mechanism isolation (ticket 04): TOST equivalence test (two one-sided
    tests) of gen6_child0(beta=15, the calibrated value) vs
    phase_decaying_ucb, margin delta = 20% of the primary
    gen6_child0(beta=15) vs hint_fixed_ucb effect size (self-referential,
    not an arbitrary external anchor — matches FDA/EMA bioequivalence and
    Lakens' TOST convention, see ticket 04's Answer for citations).

Pre-registration boundary (ticket 07): the exploratory pilot (single
domain seed=42, beta=15.0 only) that motivated this spec is NOT re-run or
re-cited here as confirmatory evidence — it only informed the beta grid's
center and the endpoint choice. Its own numbers (15/20 wins p=0.008 at
budget=20; 12/20 p=0.15 at budget=40) belong in the write-up as an
explicitly-labeled non-confirmatory preamble, not in this script's output.

Real-domain tie-back (ticket 06): out of scope. Not attempted here.

Usage:
    python run_confirmatory_spec.py                    # full spec, ~2-2.5hr
    python run_confirmatory_spec.py --n_domains 2 --n_campaigns 3   # smoke test
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import itertools
import json
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ttest_1samp

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
from excipient_campaign_mo import pareto_front_of, snap_query_mo, make_shared_inits

import torch
from tunable_synthetic_oracle import TunableSyntheticMOOracle
from full_replay import strategy_evolved_af
from af_interface import SEED_PROGRAMS

HERE = pathlib.Path(__file__).parent

# ── Fixed domain regime (ticket 02) ─────────────────────────────────────────
DOMAIN_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]  # 8 independent pool draws
PLATEAU_SHARPNESS = 5.0
NOISE_LEVEL = 0.08
NOISE_MODE = "proportional"
SCALE2 = 3.0

# ── Beta grid (ticket 03) ────────────────────────────────────────────────────
BETA_GRID = [2, 5, 10, 15, 25]
HINT_BETA = 2.0  # fixed, not swept
# Which beta in BETA_GRID gets the bootstrap-CI + TOST follow-up treatment
# (originally the pilot-calibrated beta=15; overridable via --calibrated_beta).
CALIBRATED_BETA = 15

# ── Campaign shape (unchanged convention) ────────────────────────────────────
N_CAMPAIGNS = 20
BUDGET = 40
N_INIT = 10
BATCH_SIZE = 5

SEED_HINT_FIXED_UCB_TEMPLATE = '''
def score_pool(context):
    """Fixed-weight UCB, RAW (non-front-range-normalised) sigma. beta={beta}."""
    names = context["objective_names"]
    beta = {beta}
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        scores.append(mu_sum + beta * sigma_sum)
    return scores
'''.strip("\n")

_GEN6_CHILD0_TEMPLATE = '''
def score_pool(context):
    """Front-range-normalised sigma UCB (gen6_child0). beta={beta}."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    beta = {beta}
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        scores.append(mu_sum + beta * sigma_norm)
    return scores
'''.strip("\n")


def build_conditions():
    conditions = {"hint_fixed_ucb": SEED_HINT_FIXED_UCB_TEMPLATE.format(beta=HINT_BETA)}
    for beta in BETA_GRID:
        conditions[f"gen6_child0_beta{beta}"] = _GEN6_CHILD0_TEMPLATE.format(beta=beta)
    conditions["phase_decaying_ucb"] = SEED_PROGRAMS["phase_decaying_ucb"]
    return conditions


def _hv(Y_flipped_pool, ref_point):
    """HV of a (already all-min-flipped) point set w.r.t. a fixed ref point."""
    from pymoo.indicators.hv import HV
    return float(HV(ref_point=ref_point)(Y_flipped_pool))


def compute_true_hv_and_ref(oracle):
    """Fixed reference point + true-optimum HV, both derived from the
    oracle's noiseless _Y_true pool (not the noisy _Y_raw draw) — this is
    the "true optimum" the primary endpoint's log HV difference normalises
    against (ticket 01). Ref point convention matches run_mo_campaign's
    (10% margin beyond the flipped pool's max)."""
    directions = oracle.objective_directions()
    Y_true = oracle._Y_true.copy()
    Y_flipped = Y_true.copy()
    for j, d_ in enumerate(directions):
        if d_ == "max":
            Y_flipped[:, j] = -Y_flipped[:, j]
    ref_point = (Y_flipped.max(axis=0) +
                 0.1 * (Y_flipped.max(axis=0) - Y_flipped.min(axis=0) + 1e-9))
    pf_idx = pareto_front_of(Y_true, directions=directions)
    hv_true = _hv(Y_flipped[pf_idx], ref_point)
    return hv_true, ref_point, directions


def run_one_campaign_tracked(oracle, X_init, Y_init, hv_true, ref_point, directions,
                              seed, af_code):
    """Mirrors run_mo_campaign's loop but computes HV against the TRUE-
    optimum-derived ref_point (not a noisy-pool-derived one), so
    log(hv_true - hv_observed) is well-defined and comparable across
    conditions/domain-seeds. Returns a list of per-batch hv_observed."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    bounds = oracle.bounds()

    X_obs, Y_obs = X_init.copy(), Y_init.copy()
    oracle._queried = set()
    X_all_s = oracle._scaler.transform(oracle._X_raw)
    X_init_s = oracle._scaler.transform(X_init)
    for row in X_init_s:
        idx = int(np.argmin(np.linalg.norm(X_all_s - row, axis=1)))
        oracle._queried.add(idx)

    n_batches = max(1, (BUDGET - len(X_init)) // BATCH_SIZE)
    hv_trace = []

    def _hv_observed(Y):
        Yf = Y.copy()
        for j, d_ in enumerate(directions):
            if d_ == "max":
                Yf[:, j] = -Yf[:, j]
        pf_idx = pareto_front_of(Y, directions=directions)
        return _hv(Yf[pf_idx], ref_point)

    hv_trace.append(_hv_observed(Y_obs))  # batch 0 = init-only

    for b in range(n_batches):
        try:
            candidates, _ = strategy_evolved_af(
                oracle=oracle, X_obs=X_obs, Y_obs=Y_obs, bounds=bounds,
                batch_size=BATCH_SIZE, rng=rng, af_code=af_code, budget=BUDGET)
        except Exception as e:
            d = bounds.shape[0]
            lo, hi = bounds[:, 0], bounds[:, 1]
            candidates = rng.uniform(lo, hi, (BATCH_SIZE, d))
        new_x, new_y = snap_query_mo(candidates[:BATCH_SIZE], oracle)
        if new_x:
            X_obs = np.vstack([X_obs, new_x])
            Y_obs = np.vstack([Y_obs, new_y])
        hv_trace.append(_hv_observed(Y_obs))

    return hv_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_domains", type=int, default=len(DOMAIN_SEEDS))
    ap.add_argument("--n_campaigns", type=int, default=N_CAMPAIGNS)
    ap.add_argument("--out_path", default=str(HERE / "confirmatory_spec_results.json"))
    ap.add_argument("--raw_out_path", default=str(HERE / "confirmatory_spec_raw.parquet"))
    ap.add_argument("--betas", type=str, default=None,
                     help="Comma-separated beta grid override, e.g. '1,2,3'. "
                          "Defaults to the ticket-03 grid {2,5,10,15,25}.")
    ap.add_argument("--calibrated_beta", type=float, default=None,
                     help="Which beta (must be in the grid) gets the bootstrap-CI "
                          "+ TOST follow-up. Defaults to CALIBRATED_BETA (15), or "
                          "the middle of --betas if that's not in the override grid.")
    args = ap.parse_args()

    global BETA_GRID, CALIBRATED_BETA
    if args.betas is not None:
        BETA_GRID = [float(b) if "." in b else int(b) for b in args.betas.split(",")]
    if args.calibrated_beta is not None:
        CALIBRATED_BETA = args.calibrated_beta
    elif CALIBRATED_BETA not in BETA_GRID:
        CALIBRATED_BETA = BETA_GRID[len(BETA_GRID) // 2]

    domain_seeds = DOMAIN_SEEDS[:args.n_domains]
    conditions = build_conditions()
    print(f"Domains: {domain_seeds}\nConditions: {list(conditions)}\n"
          f"n_campaigns={args.n_campaigns}, budget={BUDGET}\n")

    rows = []  # one row per (domain_seed, condition, campaign, batch)
    for domain_seed in domain_seeds:
        t_domain0 = time.perf_counter()
        oracle = TunableSyntheticMOOracle.build(
            plateau_sharpness=PLATEAU_SHARPNESS, noise_level=NOISE_LEVEL,
            noise_mode=NOISE_MODE, scale2=SCALE2, seed=domain_seed)
        hv_true, ref_point, directions = compute_true_hv_and_ref(oracle)
        inits = make_shared_inits(oracle, args.n_campaigns, N_INIT, rng_seed=domain_seed)

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
        print(f"  domain_seed={domain_seed} done in {time.perf_counter()-t_domain0:.1f}s "
              f"(hv_true={hv_true:.4g})")

    df = pd.DataFrame(rows)
    df.to_parquet(args.raw_out_path)
    print(f"\nSaved raw per-batch results to {args.raw_out_path} ({len(df)} rows)")

    # ── Primary endpoint: AUC of log_hv_diff per (domain_seed, condition, campaign) ──
    auc = (df.groupby(["domain_seed", "condition", "campaign"])["log_hv_diff"]
             .mean().reset_index().rename(columns={"log_hv_diff": "auc"}))

    # ── Secondary: batches-to-90% ────────────────────────────────────────────
    def _batches_to_90(g):
        hv_true_g = g["hv_true"].iloc[0]
        thresh = 0.9 * hv_true_g
        hit = g.loc[g["hv_observed"] >= thresh, "batch"]
        return int(hit.min()) if len(hit) else int(g["batch"].max())
    b90 = (df.groupby(["domain_seed", "condition", "campaign"])
             .apply(_batches_to_90, include_groups=False)
             .reset_index(name="batches_to_90"))

    # ── Tertiary: final HV at budget=40 ─────────────────────────────────────
    final_hv = (df.loc[df.groupby(["domain_seed", "condition", "campaign"])["batch"]
                       .idxmax()][["domain_seed", "condition", "campaign", "hv_observed"]]
                  .rename(columns={"hv_observed": "final_hv"}))

    summary = auc.merge(b90, on=["domain_seed", "condition", "campaign"]) \
                 .merge(final_hv, on=["domain_seed", "condition", "campaign"])
    summary.to_csv(str(HERE / "confirmatory_spec_summary.csv"), index=False)
    print(f"Saved per-campaign summary to confirmatory_spec_summary.csv")

    # ── Primary confirmatory test: MixedLM per beta, BH-corrected across betas ──
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("\nstatsmodels not installed — skipping MixedLM stage. "
              "pip install statsmodels and rerun stats on the saved parquet/csv.")
        return

    print(f"\n{'='*70}\nPrimary confirmatory test (ticket 05): "
          f"AUC ~ condition + (1|domain_seed), per beta\n{'='*70}")
    beta_pvals, beta_effects, beta_names = [], [], []
    hint_auc = auc[auc["condition"] == "hint_fixed_ucb"]
    for beta in BETA_GRID:
        cond_name = f"gen6_child0_beta{beta}"
        sub = pd.concat([auc[auc["condition"] == cond_name], hint_auc])
        sub = sub.copy()
        sub["is_gen6"] = (sub["condition"] == cond_name).astype(int)
        try:
            md = smf.mixedlm("auc ~ is_gen6", sub, groups=sub["domain_seed"])
            fit = md.fit()
            pval = float(fit.pvalues["is_gen6"])
            effect = float(fit.params["is_gen6"])  # negative = gen6 has LOWER (better) log-diff
        except Exception as e:
            pval, effect = float("nan"), float("nan")
            print(f"  beta={beta}: MixedLM failed ({e})")
        beta_pvals.append(pval)
        beta_effects.append(effect)
        beta_names.append(cond_name)
        print(f"  beta={beta:>3}: effect(is_gen6 on AUC)={effect:+.4f}  p={pval:.4f}")

    valid = [i for i, p in enumerate(beta_pvals) if not np.isnan(p)]
    bh_flags = [False] * len(beta_pvals)
    if valid:
        bh_sig = false_discovery_control([beta_pvals[i] for i in valid], method="bh")
        for k, i in enumerate(valid):
            bh_flags[i] = bool(bh_sig[k] < 0.05)

    same_sign = [e < 0 for e in beta_effects if not np.isnan(e)]  # negative = favors gen6
    n_same_sign = sum(same_sign)
    n_bh_sig = sum(bh_flags)
    n_grid = len(BETA_GRID)
    need_same_sign = math.ceil(0.8 * n_grid)
    need_bh_sig = math.ceil(0.6 * n_grid)
    print(f"\nRobustness (ticket 03 criterion, scaled to this grid): same-sign favoring "
          f"gen6_child0 in {n_same_sign}/{n_grid} betas (need >={need_same_sign}); "
          f"BH-significant in {n_bh_sig}/{n_grid} betas (need >={need_bh_sig}).")
    robust = (n_same_sign >= need_same_sign) and (n_bh_sig >= need_bh_sig)
    print(f"ROBUSTNESS CRITERION: {'MET' if robust else 'NOT MET'}")

    # ── Cluster bootstrap CI (domain-seed resampling) for the calibrated beta ──
    print(f"\n{'='*70}\nCluster bootstrap CI (ticket 05), gen6_child0_beta{CALIBRATED_BETA:g} vs hint_fixed_ucb")
    rng = np.random.default_rng(0)
    calibrated = f"gen6_child0_beta{CALIBRATED_BETA:g}"
    a = auc[auc["condition"] == calibrated]
    h = auc[auc["condition"] == "hint_fixed_ucb"]
    seeds_present = sorted(auc["domain_seed"].unique())
    boot_diffs = []
    for _ in range(2000):
        resample = rng.choice(seeds_present, size=len(seeds_present), replace=True)
        a_mean = np.mean([a[a["domain_seed"] == s]["auc"].mean() for s in resample])
        h_mean = np.mean([h[h["domain_seed"] == s]["auc"].mean() for s in resample])
        boot_diffs.append(a_mean - h_mean)
    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    print(f"  AUC diff ({calibrated} - hint_fixed_ucb): mean={boot_diffs.mean():+.4f}  "
          f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  (negative = gen6 better)")

    # ── TOST equivalence: calibrated gen6_child0 vs phase_decaying_ucb (ticket 04) ──
    print(f"\n{'='*70}\nMechanism isolation (ticket 04): TOST {calibrated} vs phase_decaying_ucb")
    primary_effect = float(a["auc"].mean() - h["auc"].mean())
    delta = 0.2 * abs(primary_effect)
    print(f"  Primary effect ({calibrated} vs hint_fixed_ucb) = {primary_effect:+.4f}; "
          f"TOST margin delta = 20% of |effect| = {delta:.4f}")
    p_decay = auc[auc["condition"] == "phase_decaying_ucb"]
    diff_vals = (a.groupby("domain_seed")["auc"].mean().values -
                 p_decay.groupby("domain_seed")["auc"].mean().values)
    # TOST via two one-sided t-tests against +/- delta
    t_lo = ttest_1samp(diff_vals, -delta, alternative="greater")
    t_hi = ttest_1samp(diff_vals, delta, alternative="less")
    tost_p = max(t_lo.pvalue, t_hi.pvalue)
    print(f"  mean(diff)={diff_vals.mean():+.4f}  TOST p={tost_p:.4f} "
          f"({'EQUIVALENT' if tost_p < 0.05 else 'NOT shown equivalent'} at delta={delta:.4f})")

    with open(args.out_path, "w") as f:
        json.dump({
            "domain_seeds": domain_seeds, "beta_grid": BETA_GRID,
            "beta_pvals": beta_pvals, "beta_effects": beta_effects,
            "beta_names": beta_names, "bh_flags": bh_flags,
            "n_same_sign": n_same_sign, "n_bh_sig": n_bh_sig, "robust": robust,
            "bootstrap_ci": {"mean": float(boot_diffs.mean()),
                              "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)},
            "tost": {"primary_effect": primary_effect, "delta": delta,
                     "mean_diff": float(diff_vals.mean()), "tost_p": float(tost_p)},
        }, f, indent=2)
    print(f"\nSaved analysis summary to {args.out_path}")


if __name__ == "__main__":
    main()
