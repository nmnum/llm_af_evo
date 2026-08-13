"""
run_phase3.py — Phase 3: Cross-domain generalisability on real ADA coatings
data, MULTI-OBJECTIVE (not the earlier synthetic single-objective version).

Tests whether the LS-NA-EGBO architecture generalises to a different domain
(coatings, real MacLeod et al. 2022 data) with only prompt changes — no code
changes to the strategy itself.

REVISION (2026-08-12): the original version of this script used
SyntheticCoatingsOracle, a synthetic single-objective (4D->1) oracle. mAb —
the domain this architecture was designed and validated on — is genuinely
multi-objective (Tm/kD/viscosity, Pareto front, qLogNEHVI + novelty-aware
selection). Testing generalisation on a single-objective coatings problem
required REWRITING the strategies as single-objective variants (qNEI instead
of qLogNEHVI, scalar UCB instead of Pareto-dominance ranking) — i.e. it
confounded "does the architecture transfer to a new domain" with "does a
different, SO-adapted architecture work at all", which is not the claim
Phase 3 is meant to test.

This version instead uses llm_af_evo/shared/ada_coatings_oracle.py's
DiscreteADACoatingsOracle — REAL measured data (253 samples,
github.com/berlinguette/ada, MacLeod et al. 2022, Nature Communications),
genuinely multi-objective (conductivity maximize, conductance_std minimize,
confirmed non-degenerate: 31.2% of the pool is Pareto-optimal), 4D
continuous inputs. The campaign machinery — run_mo_campaign,
strategy_mo_egbo_novelty, strategy_mo_egbo, pareto_front_of — is imported
UNCHANGED from excipient_campaign_mo.py / strategy_ls_na_egbo.py, the exact
code already used and validated on the mAb domain. Only the oracle and the
LLM prompts (this file's build_ada_coatings_prompt / MO coatings warm-start)
differ, matching the "only prompt changes, no strategy-code changes" design
goal for real.

Conditions:
  random       — strategy_mo_random (unmodified, from excipient_campaign_mo)
  egbo         — strategy_mo_egbo (unmodified)
  egbo_novelty — strategy_mo_egbo_novelty (unmodified, qLogNEHVI + UNSGA3 +
                 novelty-aware selection — the mAb domain's own EGBO)
  ls_na_egbo   — LLM warm-start (once, real ADA-domain prompt) + unmodified
                 strategy_mo_egbo_novelty for the rest of the campaign
  egbo_warmstart — LLM warm-start + strategy_mo_egbo_real (no novelty term).
                 4th cell of the {warm-start, novelty} 2x2: egbo=no/no,
                 egbo_novelty=no/yes, egbo_warmstart=yes/no, ls_na_egbo=
                 yes/yes -- isolates whether warm-start alone (vs. warm-
                 start x novelty interaction) explains ls_na_egbo's
                 recovery over egbo_novelty.
  ucb_warmstart — LLM warm-start + strategy_mo_scalarized_ucb (equal-weight
                 scalarised UCB, not the full qLogNEHVI/novelty backbone) --
                 does warm-start help with a much simpler MO acquisition?
  llm_labo     — LLM proposes raw candidates every batch, trust-weighted
                 mixing with strategy_mo_egbo candidates (mirrors mAb's own
                 mo_llm condition, just with raw x-vectors instead of named
                 formulations — coatings has no excipient catalogue)

Metrics: final hypervolume (fixed reference point, anchored to the FULL
oracle pool, same convention as run_mo_campaign's own HV reporting) as a
fraction of the full pool's true Pareto-front hypervolume (the best any
strategy could achieve given this discrete pool), and experiments-to-90%-
of-that-ceiling.

Usage:
    python run_phase3.py --mock_llm --n_seeds 20
    python run_phase3.py --model qwen3:32b --n_seeds 20

    # Restart after interruption — just re-run the same command, it resumes
    # from results/benchmark_phase3/phase3_raw_results.csv (or --out_dir).

Resumable: writes a checkpoint row to <out_dir>/phase3_raw_results.csv
after every single (condition, seed) combo, not just at the end — the
real-LLM run (llm_labo calls the LLM every batch, ls_na_egbo once per
campaign) is long enough that losing all progress to a crash or disconnect
partway through would be expensive to re-pay. On restart, any
(condition, seed) combo already present in the checkpoint is skipped.
"""

import argparse
import pathlib
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "llm_af_evo" / "shared"))

from ada_coatings_oracle import DiscreteADACoatingsOracle
from excipient_campaign_mo import (
    run_mo_campaign, make_shared_inits, pareto_front_of,
    strategy_mo_random, strategy_mo_egbo_real,
    mixing_weight, pareto_filter_candidates, _fit_gp_1d, _make_pool,
)
from strategy_ls_na_egbo import strategy_mo_egbo_novelty
from llm_warmstart import diversity_select, parse_llm_coatings, _mock_warmstart_coatings


def per_objective_trust_generic(X_obs: np.ndarray, Y_obs: np.ndarray,
                                 objective_names) -> dict:
    """Dimension-agnostic copy of excipient_campaign_mo.per_objective_trust
    (same LOO-calibration + ls/nn-ratio + uncertainty-reduction battery,
    identical math and thresholds) — that function hardcodes the module-
    level OBJECTIVE_NAMES = ["Tm","kD","viscosity"] (M=3), so calling it
    directly here would IndexError on Y_obs[:, 2] for coatings' 2-objective
    oracle. Kept as a local copy rather than patching the shared mAb-domain
    function, to avoid touching validated mAb infrastructure for a
    coatings-specific fix."""
    n, d = X_obs.shape
    scores = {}
    for j, name in enumerate(objective_names):
        y = Y_obs[:, j]
        if n < 3:
            scores[name] = 0.5
            continue

        loo_z = []
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            if mask.sum() < 2:
                continue
            gp_i, sc_i = _fit_gp_1d(X_obs[mask], y[mask])
            mu, sigma = gp_i.predict(sc_i.transform(X_obs[i:i+1]), return_std=True)
            sigma = max(float(sigma[0]), 1e-6)
            loo_z.append((float(mu[0]) - y[i]) / sigma)
        loo_mean_abs_z = float(np.mean(np.abs(loo_z))) if loo_z else 1.0
        loo_score = (1.0 if 0.3 <= loo_mean_abs_z <= 1.8 else
                     max(0.0, 1.0 - (loo_mean_abs_z - 1.8) / 2.0) if loo_mean_abs_z > 1.8
                     else 0.6)

        obs_per_dim = n / d
        gp_full, sc_full = _fit_gp_1d(X_obs, y)
        ls = float(gp_full.kernel_.length_scale)
        Xn = sc_full.transform(X_obs)
        nn_dists = []
        for i in range(n):
            dists = np.linalg.norm(Xn - Xn[i], axis=1)
            dists[i] = np.inf
            nn_dists.append(dists.min())
        mean_nn = float(np.mean(nn_dists))
        ls_nn_ratio = ls / (mean_nn + 1e-12)

        if obs_per_dim < 2.0 or abs(ls - 1e-3) < 1e-4:
            ls_score = 0.5
        elif ls_nn_ratio < 0.8:
            ls_score = 0.0
        elif ls_nn_ratio > 8.0:
            ls_score = 0.5
        else:
            ls_score = 1.0

        rng_test = np.random.default_rng(0)
        test_pts = rng_test.random((100, d))
        _, sigma_post = gp_full.predict(sc_full.transform(test_pts), return_std=True)
        uncertainty_reduction = float(np.clip(1.0 - sigma_post.mean(), -1, 1))
        unc_score = 0.0 if uncertainty_reduction > 0.3 and loo_mean_abs_z > 1.8 else \
                    (0.5 if uncertainty_reduction < 0.1 else 1.0)

        if obs_per_dim < 2.0:
            trust = 0.6 * loo_score + 0.15 * ls_score + 0.25 * unc_score
        else:
            trust = 0.5 * loo_score + 0.3 * ls_score + 0.2 * unc_score
        scores[name] = float(np.clip(trust, 0, 1))

    return scores


CKPT_COLS = [
    "phase", "condition", "seed", "budget", "n_init", "final_hv",
    "hv_frac", "exp_to_90pct_hv", "n_obs", "final_pf_size",
    "llm_fallback_used", "llm_retry_count",
]


def load_checkpoint(ckpt_path):
    """Load existing checkpoint CSV, return DataFrame or empty. Same pattern
    as run_benchmark_resumable.py's load_checkpoint."""
    if pathlib.Path(ckpt_path).exists():
        try:
            df = pd.read_csv(ckpt_path)
            print(f"  Loaded checkpoint: {len(df)} rows from {ckpt_path}")
            return df
        except Exception as e:
            print(f"  Warning: could not load checkpoint: {e}")
    return pd.DataFrame(columns=CKPT_COLS)


def save_checkpoint(df, ckpt_path):
    df.to_csv(str(ckpt_path), index=False)


def is_completed(ckpt_df, condition, seed):
    if len(ckpt_df) == 0:
        return False
    mask = (ckpt_df["condition"] == condition) & (ckpt_df["seed"] == seed)
    return mask.any()


def compute_pool_hv(oracle):
    """Hypervolume of the FULL oracle pool's true Pareto front — the best
    any strategy could achieve given this discrete 253-point dataset. Uses
    the exact same fixed-reference-point formula as run_mo_campaign's own
    HV reporting, so every condition's final_hv is comparable against this
    ceiling on a consistent scale."""
    from pymoo.indicators.hv import HV
    directions = oracle.objective_directions()
    Y_all = oracle._Y_raw.copy()
    Y_all_flipped = Y_all.copy()
    for j, d_ in enumerate(directions):
        if d_ == "max":
            Y_all_flipped[:, j] = -Y_all_flipped[:, j]
    fixed_ref = (Y_all_flipped.max(axis=0) +
                 0.1 * (Y_all_flipped.max(axis=0) - Y_all_flipped.min(axis=0) + 1e-9))
    pf_idx = pareto_front_of(Y_all, directions=directions)
    Y = Y_all.copy()
    for j, d_ in enumerate(directions):
        if d_ == "max":
            Y[:, j] = -Y[:, j]
    return float(HV(ref_point=fixed_ref)(Y[pf_idx]))


# ── Real-ADA-domain LLM prompt (NOT the synthetic concentration/temperature/
# flow_rate/pressure description from llm_warmstart.py's LAYER3_COATINGS —
# that describes SyntheticCoatingsOracle's made-up physical meaning, which
# does not match DiscreteADACoatingsOracle's real inputs/objectives) ────────

def build_ada_coatings_prompt(n_propose: int, bounds: np.ndarray) -> str:
    lo, hi = bounds[:, 0], bounds[:, 1]
    dims = [
        ("x0", "fuel to oxidizer ratio", lo[0], hi[0]),
        ("x1", "acac amount (glycine <-> acetylacetone composition)", lo[1], hi[1]),
        ("x2", "total precursor concentration (g/mL)", lo[2], hi[2]),
        ("x3", "annealing temperature (Celsius)", lo[3], hi[3]),
    ]
    dim_lines = "\n".join(
        f"  {name} ({desc}): observed range {v_lo:.4g} to {v_hi:.4g}"
        for name, desc, v_lo, v_hi in dims
    )
    return f"""You are optimising a solution-combustion-synthesis thin-film coating
process with 4 continuous input parameters, tracking TWO separate objectives
simultaneously:
  conductivity      (S/m, higher = better)
  conductance_std   (Siemens, lower = better — within-sample uniformity;
                     higher position-to-position variability means a less
                     uniform film)
There is no single "best" input combination — you are looking for good
TRADE-OFFS across these two objectives (a Pareto front), not one optimal
point.

Input parameters (real experimental units — NOT normalised):
{dim_lines}

Propose exactly {n_propose} new candidate points to evaluate. For each,
report x AS A FRACTION of the observed range above (0=minimum observed,
1=maximum observed) — i.e. x0=0.5 means the midpoint of the fuel:oxidizer
ratio range, not the raw ratio itself.

Respond with valid JSON only:
{{
  "candidates": [
    {{"x": [x0, x1, x2, x3], "targets": ["conductivity"|"conductance_std", ...],
      "tradeoff": "<short phrase>"}}
  ]
}}"""


def llm_warmstart_init_coatings_mo(oracle, n_propose=25, n_select=10,
                                    model="qwen3:32b", mock=False, seed=42):
    """MO analogue of llm_warmstart.llm_warmstart_init_coatings — returns
    (X_init, Y_init) with Y_init a (n_select, M) matrix (this oracle's real
    multi-objective readout), not a scalar. Uses the real ADA-domain
    prompt above instead of the generic SyntheticCoatingsOracle one."""
    d = oracle.bounds().shape[0]
    bounds = oracle.bounds()

    retry_count = 0
    fallback_used = False

    if mock:
        candidates_n = _mock_warmstart_coatings(n_propose, seed, d)
    else:
        prompt = build_ada_coatings_prompt(n_propose, bounds)
        import ollama
        candidates_n = np.array([])
        for attempt in range(3):
            retry_count = attempt
            try:
                resp = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system",
                         "content": "You are an expert optimisation assistant. "
                                    "Respond with valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.4, "num_predict": 1024, "think": False,
                              "seed": seed + attempt},
                )
                text = resp["message"]["content"]
                candidates_n = parse_llm_coatings(text, n_propose, d)
                if len(candidates_n) > 0:
                    break
            except Exception:
                if attempt == 2:
                    fallback_used = True
                    candidates_n = _mock_warmstart_coatings(n_propose, seed, d)

    if len(candidates_n) == 0:
        fallback_used = True
        candidates_n = _mock_warmstart_coatings(n_propose, seed, d)

    selected_idx = diversity_select(candidates_n, n_select)
    selected_n = candidates_n[selected_idx]

    lo, hi = bounds[:, 0], bounds[:, 1]
    selected_raw = selected_n * (hi - lo) + lo

    all_X = oracle._X_raw
    scaler = oracle._scaler
    X_all_s = scaler.transform(all_X)
    M = oracle._Y_raw.shape[1]

    X_init = np.zeros((len(selected_idx), d))
    Y_init = np.zeros((len(selected_idx), M))
    queried = set()
    for i, cand in enumerate(selected_raw):
        cand_s = scaler.transform(cand.reshape(1, -1))[0]
        unqueried = [j for j in range(len(all_X)) if j not in queried]
        if not unqueried:
            unqueried = list(range(len(all_X)))
        pool_s = scaler.transform(all_X[unqueried])
        chosen = unqueried[int(np.argmin(np.linalg.norm(pool_s - cand_s, axis=1)))]
        queried.add(chosen)
        X_init[i] = all_X[chosen]
        Y_init[i] = oracle._Y_raw[chosen]

    meta = {"llm_fallback_used": fallback_used, "llm_retry_count": retry_count}
    return X_init, Y_init, meta


def run_ls_na_egbo_campaign_coatings(oracle, budget, strategy_fn, strategy_kwargs,
                                      batch_size=5, seed=0, n_init=10, n_propose=25,
                                      model="qwen3:32b", mock_llm=False):
    """MO coatings analogue of strategy_ls_na_egbo.run_ls_na_egbo_campaign:
    Stage 1 LLM warm-start (real ADA prompt), Stage 2 unmodified
    strategy_fn (strategy_mo_egbo_novelty) via run_mo_campaign — same
    two-stage architecture, same Stage-2 code, only Stage 1's oracle/prompt
    differ from the mAb version."""
    X_init, Y_init, meta = llm_warmstart_init_coatings_mo(
        oracle, n_propose=n_propose, n_select=n_init,
        model=model, mock=mock_llm, seed=seed,
    )

    oracle._queried = set()
    X_all_s = oracle._scaler.transform(oracle._X_raw)
    for row in oracle._scaler.transform(X_init):
        idx = int(np.argmin(np.linalg.norm(X_all_s - row, axis=1)))
        oracle._queried.add(idx)

    result = run_mo_campaign(
        oracle, X_init, Y_init, budget, strategy_fn, strategy_kwargs,
        batch_size=batch_size, seed=seed,
    )
    result["llm_fallback_used"] = meta["llm_fallback_used"]
    result["llm_retry_count"] = meta["llm_retry_count"]
    return result


def strategy_mo_llm_coatings(oracle, X_obs, Y_obs, bounds, batch_size, rng,
                              model="qwen3:32b", mock_llm=False,
                              n_llm_candidates=20, **kw):
    """LLM-every-batch condition for coatings — same trust-weighted
    LLM/GP mixing architecture as excipient_campaign_mo.strategy_mo_llm
    (mAb's own 'LLM every batch' condition), just proposing raw x-vectors
    via build_ada_coatings_prompt instead of named formulations (coatings
    has no excipient catalogue to name)."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    trust = per_objective_trust_generic(X_obs, Y_obs, oracle.objective_names())
    is_sparse = (len(X_obs) / bounds.shape[0]) < 2.0
    weight = mixing_weight(trust, sparse=is_sparse)

    n_llm_propose = max(batch_size, int(n_llm_candidates * (1.5 - weight)))

    if mock_llm:
        rng_mock = np.random.default_rng(rng.integers(1_000_000))
        d = bounds.shape[0]
        llm_cands_n = np.zeros((n_llm_propose, d))
        for j in range(d):
            perm = rng_mock.permutation(n_llm_propose)
            llm_cands_n[:, j] = (perm + rng_mock.uniform(0, 1, n_llm_propose)) / n_llm_propose
    else:
        prompt = build_ada_coatings_prompt(n_llm_propose, bounds)
        import ollama
        llm_cands_n = np.array([])
        for attempt in range(3):
            try:
                resp = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system",
                         "content": "You are an expert optimisation assistant. "
                                    "Respond with valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.4, "num_predict": 1024, "think": False,
                              "seed": int(rng.integers(1_000_000)) + attempt},
                )
                text = resp["message"]["content"]
                llm_cands_n = parse_llm_coatings(text, n_llm_propose, bounds.shape[0])
                if len(llm_cands_n) > 0:
                    break
            except Exception:
                if attempt == 2:
                    llm_cands_n = rng.random((n_llm_propose, bounds.shape[0]))

    llm_raw = (llm_cands_n * (hi - lo) + lo if len(llm_cands_n) > 0
               else np.zeros((0, len(lo))))

    gp_cands, _ = strategy_mo_egbo_real(oracle, X_obs, Y_obs, bounds, batch_size, rng)
    pool_raw = np.vstack([llm_raw, gp_cands]) if len(llm_raw) else gp_cands

    n_llm_pre_filter = len(llm_raw)
    source_is_llm = np.zeros(len(pool_raw), dtype=bool)
    source_is_llm[:n_llm_pre_filter] = True

    X_obs_n = (X_obs - lo) / (hi - lo + 1e-12)
    pool_n = (pool_raw - lo) / (hi - lo + 1e-12)
    keep = pareto_filter_candidates(pool_n, X_obs_n, None)
    if len(keep) > 0:
        pool_raw = pool_raw[keep]
        source_is_llm = source_is_llm[keep]

    n_select = min(batch_size, len(pool_raw))
    n_llm_in_pool = int(source_is_llm.sum())
    n_gp_in_pool = len(pool_raw) - n_llm_in_pool

    if n_llm_in_pool > 0 and n_gp_in_pool > 0:
        gp_prob_each = weight / n_gp_in_pool
        llm_prob_each = (1.0 - weight) / n_llm_in_pool
        probs = np.where(source_is_llm, llm_prob_each, gp_prob_each)
        probs = probs / probs.sum()
        selected_idx = rng.choice(len(pool_raw), n_select, replace=False, p=probs)
    else:
        selected_idx = (rng.choice(len(pool_raw), n_select, replace=False)
                         if len(pool_raw) > n_select else np.arange(len(pool_raw)))

    return pool_raw[selected_idx], {
        "labo": True, "trust": trust, "mixing_weight": weight,
        "n_llm_proposed": len(llm_cands_n), "n_llm_in_pool": n_llm_in_pool,
    }


def strategy_mo_scalarized_ucb(oracle, X_obs, Y_obs, bounds, batch_size, rng,
                                beta=2.0, **kw):
    """Baseline: scalarised (equal-weight-sum) UCB acquisition across all
    objectives, ranked and truncated to batch_size — the classic single-
    scalar-acquisition MO baseline, in contrast to egbo/egbo_novelty/
    egbo_real's Pareto-dominance or qLogNEHVI-based selection. Per-objective
    GP mean/std are standardised by that objective's observed std before
    summing, so raw-unit mismatches (S/m conductivity vs. Siemens
    conductance_std) don't let one objective dominate the scalarisation.
    Dimension-agnostic (uses oracle.objective_directions() and
    Y_obs.shape[1] dynamically) -- local to this file rather than added to
    excipient_campaign_mo.py since it's coatings-specific, added for the
    warm-start-with-a-simpler-acquisition ablation."""
    directions = oracle.objective_directions()
    M = Y_obs.shape[1]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X_obs_n = (X_obs - lo) / (hi - lo + 1e-12)
    pool_n = _make_pool(X_obs_n, Y_obs, bounds, rng, n=60, directions=directions)
    pool_raw = pool_n * (hi - lo) + lo

    scores = np.zeros(len(pool_raw))
    for j in range(M):
        sign = 1.0 if directions[j] == "max" else -1.0
        y_sig = sign * Y_obs[:, j]
        y_std = float(np.std(y_sig)) or 1.0
        gp, sc = _fit_gp_1d(X_obs, y_sig)
        mu, sigma = gp.predict(sc.transform(pool_raw), return_std=True)
        scores += (mu + beta * sigma) / y_std

    top_idx = np.argsort(scores)[-batch_size:]
    return pool_raw[top_idx], {"scalarized_ucb": True}


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Coatings generalisability benchmark (real MO data)")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--model", default="qwen3:32b")
    parser.add_argument("--mock_llm", action="store_true")
    parser.add_argument("--w_acq", type=float, default=0.9)
    parser.add_argument("--w_nov", type=float, default=0.1)
    parser.add_argument("--out_dir", default="results/benchmark_phase3")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "phase3_raw_results.csv"

    print(f"\n{'='*60}")
    print(f"PHASE 3: Coatings generalisability (real ADA data, multi-objective)")
    print(f"{'='*60}")
    print(f"  Seeds: {args.n_seeds}, Budget: {args.budget}")
    print(f"  N_init: {args.n_init}, Batch: {args.batch_size}")
    print(f"  Mock LLM: {args.mock_llm}")

    oracle = DiscreteADACoatingsOracle.build()
    print(f"  Oracle: {len(oracle)} real samples, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")
    pool_hv = compute_pool_hv(oracle)
    print(f"  Full-pool Pareto-front HV (ceiling): {pool_hv:.4g}")

    shared_inits = make_shared_inits(oracle, args.n_seeds, args.n_init, rng_seed=42)

    conditions = {
        "random": (strategy_mo_random, {}, False),
        "egbo": (strategy_mo_egbo_real, {}, False),
        "egbo_novelty": (strategy_mo_egbo_novelty,
                          {"w_acq": args.w_acq, "w_nov": args.w_nov}, False),
        "ls_na_egbo": (strategy_mo_egbo_novelty,
                        {"w_acq": args.w_acq, "w_nov": args.w_nov}, True),
        # 4th cell of the 2x2 {warm-start, novelty} design -- egbo (no
        # novelty) + LLM warm-start, isolating whether the ls_na_egbo
        # recovery over egbo_novelty is warm-start alone or a warm-start x
        # novelty interaction.
        "egbo_warmstart": (strategy_mo_egbo_real, {}, True),
        # LLM warm-start + a simpler scalarised-UCB acquisition instead of
        # the full qLogNEHVI/novelty EGBO backbone -- does warm-start help
        # independent of a strong MO acquisition strategy?
        "ucb_warmstart": (strategy_mo_scalarized_ucb, {"beta": 2.0}, True),
        "llm_labo": (strategy_mo_llm_coatings,
                      {"mock_llm": args.mock_llm, "model": args.model,
                       "n_llm_candidates": 20}, False),
    }

    ckpt_df = load_checkpoint(ckpt_path)
    results = ckpt_df.to_dict("records")
    total_combos = len(conditions) * args.n_seeds
    print(f"  Progress: {len(results)}/{total_combos} combos completed")

    t_start = time.time()
    directions = oracle.objective_directions()

    for cond_name, (fn, kwargs, is_warmstart) in conditions.items():
        t0 = time.time()
        pending = [s for s in range(args.n_seeds)
                   if not is_completed(ckpt_df, cond_name, s)]
        skipped = args.n_seeds - len(pending)
        print(f"\n  {cond_name}... ({skipped} already done, {len(pending)} to run)",
              end="", flush=True)

        for seed_idx in pending:
            if is_warmstart:
                result = run_ls_na_egbo_campaign_coatings(
                    oracle, args.budget, fn, kwargs,
                    batch_size=args.batch_size, seed=seed_idx, n_init=args.n_init,
                    n_propose=25, model=args.model, mock_llm=args.mock_llm,
                )
            else:
                X_init, Y_init = shared_inits[seed_idx]
                result = run_mo_campaign(
                    oracle, X_init, Y_init, args.budget, fn, kwargs,
                    batch_size=args.batch_size, seed=seed_idx,
                )

            final_hv = result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")
            hv_frac = final_hv / pool_hv if pool_hv else float("nan")

            exp_to_90 = args.budget
            for i, hv in enumerate(result["hv_trajectory"]):
                if hv >= 0.9 * pool_hv:
                    exp_to_90 = min(args.budget, args.n_init + (i + 1) * args.batch_size)
                    break

            final_pf_size = len(pareto_front_of(result["Y_obs"], directions=directions))

            results.append({
                "phase": 3, "condition": cond_name, "seed": seed_idx,
                "budget": args.budget, "n_init": args.n_init,
                "final_hv": final_hv, "hv_frac": hv_frac,
                "exp_to_90pct_hv": exp_to_90,
                "n_obs": len(result["Y_obs"]),
                "final_pf_size": final_pf_size,
                "llm_fallback_used": result.get("llm_fallback_used", False),
                "llm_retry_count": result.get("llm_retry_count", 0),
            })
            # Checkpoint after every single (condition, seed) combo — see
            # module docstring. ckpt_df is intentionally NOT refreshed from
            # this save within the loop (matches run_benchmark_resumable.py's
            # pattern) -- is_completed() is only consulted once per
            # condition above, so this is safe within a single process run;
            # a concurrent second process against the same out_dir is not
            # supported.
            save_checkpoint(pd.DataFrame(results), ckpt_path)
            print(".", end="", flush=True)
        print(f" ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t_start
    print(f"\n\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    df = pd.DataFrame(results)

    summary = df.groupby("condition").agg(
        hv_mean=("final_hv", "mean"),
        hv_std=("final_hv", "std"),
        frac_mean=("hv_frac", "mean"),
        frac_std=("hv_frac", "std"),
        exp90_mean=("exp_to_90pct_hv", "mean"),
        exp90_std=("exp_to_90pct_hv", "std"),
        pf_mean=("final_pf_size", "mean"),
    ).sort_values("hv_mean", ascending=False)
    summary.to_csv(out_dir / "phase3_summary.csv")

    print(f"\n{'='*60}")
    print(f"PHASE 3 RESULTS")
    print(f"{'='*60}")
    print(f"  {'Condition':<20} {'HV mean±std':>15} {'Frac of ceiling':>16} {'Exp→90%':>10} {'PF size':>8}")
    print(f"  {'-'*75}")
    for cond, row in summary.iterrows():
        print(f"  {cond:<20} {row['hv_mean']:>8.4g}±{row['hv_std']:<5.4g} "
              f"{row['frac_mean']:>10.1%}±{row['frac_std']:<4.1%} "
              f"{row['exp90_mean']:>8.1f}±{row['exp90_std']:<4.1f} "
              f"{row['pf_mean']:>8.1f}")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
