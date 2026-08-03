"""
excipient_campaign_mo.py — Multi-objective campaign runner for the excipient oracle.

Implements the three conditions needed to re-run the stress-test ablation
(mechanism / blank / wrong-prior) in true multi-objective form, on top of
excipient_oracle_mo.py (Tm, kD, viscosity — three separate, unweighted
objectives, no hard constraints).

Architecture, matching every resolved design decision:
  - Per-objective GPs (one per objective), never a single scalarised GP.
  - Per-objective GP-trust diagnostic (LOO calibration + ls/nn ratio +
    uncertainty reduction, same battery as the single-objective
    gp_trust_check.py, run independently per objective).
  - Mixing weight = MIN trust score across the three objectives (not
    average, not importance-weighted) — the least-trustworthy objective's
    GP determines how much to lean on it, per the earlier design decision.
  - No scalar objective weights anywhere. Candidate selection uses Pareto
    dominance + crowding, not a weighted sum.
  - GP/Pareto-bookkeeping role: track the running non-dominated set,
    filter out candidates dominated by or duplicate-with existing
    observations. NOT used as the primary acquisition engine when trust is
    low — exactly the antibody campaign's expected regime (n<=30, ~16D).
  - LLM proposes NAMED formulations (not raw coordinates), exact
    concentration match to what gets "tested" (server-side snap to nearest
    oracle pool point, never silently rounding away from what was proposed
    without logging it).
  - Structured output only: {"aa":..., "targets": [...], "tradeoff": "..."}
    — no free-text reasoning paragraph as the primary interface (kept only
    as an optional field, matching the earlier explainability decision).
  - Efficient: per-objective GPs share one StandardScaler fit; LLM calls are
    cached by observation-hash; batch pool candidates are vectorised
    wherever the Pareto/dominance/duplicate logic allows it.

Conditions:
  mo_egbo   — per-objective GP + Pareto-dominance-based candidate ranking,
              no LLM at all. The multi-objective analogue of single-objective
              EGBO's evolutionary-candidates + UCB.
  mo_llm    — LLM proposes named formulations; per-objective GP trust score
              sets how much weight LLM proposals get vs. GP/evolutionary
              proposals in the final candidate pool (this is the general,
              computed version of "LLM does almost everything" for the real
              campaign's low-trust regime).
  mo_random — pure random sampling, floor baseline.

Usage:
    # Quick mock test, no LLM needed
    python excipient_campaign_mo.py --mock_llm --n_repeats 3 --budget 30

    # Real run with stress-test-style prior levels
    python excipient_campaign_mo.py --model qwen2.5:72b-instruct \\
        --protein mAb_aggregation --prior_level L1 --n_repeats 15 --budget 40
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from excipient_oracle_mo import (
    MultiObjectiveExcipientOracle, DiscreteMOExcipientOracle,
    TM_RANGE, KD_RANGE, VISC_RANGE,
)
from excipient_oracle import (
    AMINO_ACIDS, SUGARS, SURFACTANTS, AA_LIST, SUGAR_LIST, SF_LIST,
    formulation_to_vector, vector_to_formulation, FEATURE_DIM,
)


OBJECTIVE_NAMES = ["Tm", "kD", "viscosity"]
OBJECTIVE_DIRECTIONS = ["max", "max", "min"]


# ── Prior knowledge, mirroring the single-objective stress test's L1/L4/WRONG design ──

MO_PRIORS = {

"agg_L1": """
You are optimising a protein formulation for an aggregation-prone monoclonal
antibody. You are tracking THREE separate objectives simultaneously:
  Tm (melting temperature, higher=better, conformational stability)
  kD (diffusion interaction parameter, higher=better, colloidal stability)
  viscosity (lower=better, manufacturability)
There is no single "best" formulation — you are looking for good TRADE-OFFS
across these three objectives (a Pareto front), not one optimal point.

Excipient mechanisms:
  Arginine: prevents aggregation via charge-shielding. Primarily affects kD;
    has a smaller effect on Tm. High concentrations can raise viscosity.
  Sucrose / trehalose: stabilise via preferential exclusion, primarily
    affecting Tm; largely independent of kD. Raise viscosity at high
    concentration.
  Polysorbate 80: protects against interface aggregation; synergises with
    arginine specifically for kD. Minimal effect on Tm or viscosity at
    typical concentrations.
  Methionine, EDTA: address oxidation, not this protein's dominant pathway.
    Expect limited benefit on any of the three objectives here.
  Mannitol above ~40 mM: crystallisation risk — generally avoid.

Higher total excipient concentration (amino acid + sugar combined) tends to
increase viscosity — there is a genuine trade-off between stabilisation and
manufacturability.
""",

"oxid_L1": """
You are optimising a protein formulation for an oxidation-prone monoclonal
antibody. You are tracking THREE separate objectives simultaneously:
  Tm (melting temperature, higher=better)
  kD (diffusion interaction parameter, higher=better)
  viscosity (lower=better)
There is no single "best" formulation — look for good TRADE-OFFS (a Pareto
front), not one optimal point.

Excipient mechanisms:
  Methionine: sacrificial antioxidant for oxidation-prone proteins. This
    protein's dominant pathway is oxidation, not aggregation, so methionine
    is expected to matter more here than arginine would.
  EDTA: chelates metal ions catalysing oxidation. Synergises with methionine.
  Sucrose / trehalose: stabilise Tm via preferential exclusion, largely
    independent of kD.
  Arginine and polysorbate80's aggregation-specific synergy is LESS relevant
    here since aggregation is not this protein's dominant degradation
    pathway — expect a smaller kD benefit from arginine than you would see
    on an aggregation-prone protein.
  Higher total excipient concentration tends to raise viscosity.
""",

"blank": """
You are optimising a protein formulation. You are tracking THREE separate
objectives simultaneously:
  Tm (melting temperature, higher=better)
  kD (diffusion interaction parameter, higher=better)
  viscosity (lower=better)
There is no single "best" formulation — look for good trade-offs across
these three objectives, not one optimal point. No further domain knowledge
is provided; explore the space systematically.
""",

}


# ── LLM candidate proposal (named formulations, structured tags) ──────────────

_MO_LLM_CACHE: dict = {}


def _obs_hash(X_obs, Y_obs):
    arr = np.column_stack([X_obs, Y_obs])
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]


def _fmt_formulation(form: dict, Y: np.ndarray = None) -> str:
    s = (f"{form['aa']} {form['aa_conc']:.1f}mM | {form['sugar']} "
         f"{form['sugar_conc']:.0f}mM | {form['surfactant']} "
         f"{form['surfactant_conc']:.1f}% | EDTA={'yes' if form.get('edta') else 'no'}")
    if Y is not None:
        s += f"  ->  Tm={Y[0]:.1f} kD={Y[1]:+.1f} visc={Y[2]:.1f}"
    return s


def llm_propose_mo_formulations(X_obs, Y_obs, prior_text, n_propose,
                                model, mock, rng, cache_key=None,
                                max_retries=3):
    """
    Ask the LLM to propose formulations with structured tags (not free-text
    reasoning as the primary interface, per the explainability decision).
    Returns (list_of_{formulation, targets, tradeoff}, cache_hit).
    """
    global _MO_LLM_CACHE
    if cache_key and cache_key in _MO_LLM_CACHE:
        return _MO_LLM_CACHE[cache_key], True

    aa_opts  = ", ".join(f"{a}({AMINO_ACIDS[a]['min']}-{AMINO_ACIDS[a]['max']}mM)"
                         for a in AA_LIST)
    sug_opts = ", ".join(f"{s}({SUGARS[s]['min']}-{SUGARS[s]['max']}mM)"
                         for s in SUGAR_LIST)
    sf_opts  = ", ".join(f"{sf}({SURFACTANTS[sf]['min']}-{SURFACTANTS[sf]['max']}%)"
                         for sf in SF_LIST)

    # Show current non-dominated (Pareto) formulations, not just "top-5 by
    # score" — there is no single score in multi-objective mode.
    if len(Y_obs) > 0:
        Y = Y_obs.copy()
        for j, d in enumerate(OBJECTIVE_DIRECTIONS):
            if d == "min":
                Y[:, j] = -Y[:, j]
        ge = (Y[None, :, :] >= Y[:, None, :]).all(axis=2)
        gt = (Y[None, :, :] >  Y[:, None, :]).any(axis=2)
        dominated = (ge & gt).any(axis=1)
        pareto_idx = np.where(~dominated)[0]
    else:
        pareto_idx = np.array([], dtype=int)

    obs_lines = []
    for i in pareto_idx[:8]:  # cap prompt length
        form = vector_to_formulation(X_obs[i])
        obs_lines.append(f"  {_fmt_formulation(form, Y_obs[i])}  [Pareto]")

    prompt = f"""{prior_text.strip()}

Current Pareto front ({len(pareto_idx)} non-dominated formulations so far):
{chr(10).join(obs_lines) if obs_lines else "  (none yet — this is the first batch)"}

Available excipients:
  Amino acids (choose one): {aa_opts}
  Sugars (choose one): {sug_opts}
  Surfactants (choose one): {sf_opts}
  EDTA: true or false

Propose exactly {n_propose} new formulations to test next. For each, state
which objective(s) it primarily targets and any trade-off it makes — keep
these SHORT (a few words each), not a paragraph.

Respond with JSON only:
{{
  "candidates": [
    {{"aa": "<name>", "aa_conc": <mM>, "sugar": "<name>", "sugar_conc": <mM>,
      "surfactant": "<name>", "surfactant_conc": <pct>, "edta": <true/false>,
      "targets": ["Tm"|"kD"|"viscosity", ...],
      "tradeoff": "<short phrase, e.g. 'raises viscosity slightly'>"}}
  ]
}}"""

    def snap(val, props):
        val = float(val)
        val = round(val / props["step"]) * props["step"]
        return float(np.clip(val, props["min"], props["max"]))

    def best_match(s, opts):
        s = str(s).lower().replace(" ", "").replace("-", "")
        for o in opts:
            if o.replace(" ", "") in s or s in o.replace(" ", ""):
                return o
        return opts[0]

    if mock:
        # Deterministic mock: perturb current Pareto points, or random if none yet
        proposals = []
        for _ in range(n_propose):
            if len(pareto_idx) > 0:
                base_i = pareto_idx[rng.integers(len(pareto_idx))]
                base = vector_to_formulation(X_obs[base_i])
            else:
                aa = str(rng.choice(AA_LIST))
                base = {"aa": aa, "aa_conc": AMINO_ACIDS[aa]["optimal_conc"],
                       "sugar": str(rng.choice(SUGAR_LIST)),
                       "sugar_conc": 50.0,
                       "surfactant": str(rng.choice(SF_LIST)),
                       "surfactant_conc": 0.4, "edta": bool(rng.random() > 0.5)}
            aa_p = AMINO_ACIDS[base["aa"]]
            base = dict(base)
            base["aa_conc"] = snap(base["aa_conc"] + rng.normal(0, aa_p["breadth"]*0.3),
                                   aa_p)
            proposals.append({"formulation": base, "targets": ["Tm", "kD"],
                              "tradeoff": "mock perturbation"})
        if cache_key:
            _MO_LLM_CACHE[cache_key] = proposals
        return proposals, False

    import ollama
    text = ""
    for attempt in range(max_retries):
        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a pharmaceutical formulation scientist. "
                                "Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.2, "num_predict": 1024, "think": False},
            )
            text = resp["message"]["content"]
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            text = re.sub(r'```(?:json)?', '', text).strip().strip('`')
            text = re.sub(r',(\s*[}\]])', r'\1', text)
            parsed = json.loads(text)
            raw = parsed.get("candidates", [])

            proposals = []
            for c in raw:
                try:
                    aa  = best_match(c.get("aa", ""), AA_LIST)
                    sug = best_match(c.get("sugar", ""), SUGAR_LIST)
                    sf  = best_match(c.get("surfactant", ""), SF_LIST)
                    form = {
                        "aa": aa,
                        "aa_conc": snap(c.get("aa_conc", AMINO_ACIDS[aa]["optimal_conc"]),
                                       AMINO_ACIDS[aa]),
                        "sugar": sug,
                        "sugar_conc": snap(c.get("sugar_conc", SUGARS[sug]["optimal_conc"]),
                                          SUGARS[sug]),
                        "surfactant": sf,
                        "surfactant_conc": snap(c.get("surfactant_conc",
                                                      SURFACTANTS[sf]["optimal_conc"]),
                                                SURFACTANTS[sf]),
                        "edta": bool(c.get("edta", False)),
                    }
                    proposals.append({
                        "formulation": form,
                        "targets": c.get("targets", []),
                        "tradeoff": c.get("tradeoff", ""),
                    })
                except Exception:
                    continue

            if proposals:
                if cache_key:
                    _MO_LLM_CACHE[cache_key] = proposals
                return proposals[:n_propose], False

        except Exception as e:
            if attempt == max_retries - 1:
                warnings.warn(f"LLM MO proposal failed: {e} | resp: {text[:150]}")

    return [], False


# ── Per-objective GP + trust diagnostic ────────────────────────────────────────

def _fit_gp_1d(X_obs, y_obs):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xs = sc.fit_transform(X_obs)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
        alpha=1e-6, normalize_y=True, n_restarts_optimizer=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(Xs, y_obs)
    return gp, sc


def per_objective_trust(X_obs: np.ndarray, Y_obs: np.ndarray) -> dict:
    """
    Run the LOO-calibration + ls/nn-ratio + uncertainty-reduction battery
    independently per objective. Returns {objective_name: trust_score}.
    Reuses the same diagnostic logic and thresholds as gp_trust_check.py
    (single-objective version), applied once per column of Y_obs.
    """
    n, d = X_obs.shape
    scores = {}

    for j, name in enumerate(OBJECTIVE_NAMES):
        y = Y_obs[:, j]
        if n < 3:
            scores[name] = 0.5
            continue

        # LOO calibration
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

        # ls/nn ratio (skip if too sparse, per the earlier dimensionality fix)
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
            ls_score = 0.5  # sparse — neutral, per the dimensionality-aware fix
        elif ls_nn_ratio < 0.8:
            ls_score = 0.0
        elif ls_nn_ratio > 8.0:
            ls_score = 0.5
        else:
            ls_score = 1.0

        # Uncertainty reduction
        rng_test = np.random.default_rng(0)
        test_pts = rng_test.random((100, d))
        _, sigma_post = gp_full.predict(sc_full.transform(test_pts), return_std=True)
        uncertainty_reduction = float(np.clip(1.0 - sigma_post.mean(), -1, 1))
        unc_score = 0.0 if uncertainty_reduction > 0.3 and loo_mean_abs_z > 1.8 else \
                    (0.5 if uncertainty_reduction < 0.1 else 1.0)

        # Measurement-noise component (fixes the collapse-to-identical-trust
        # bug): fit a SEPARATE GP with an explicit WhiteKernel noise term so
        # the model can actually attribute some variance to measurement
        # noise rather than being forced to explain everything via the
        # signal kernel. This is the same pattern already validated in
        # oracle_gap_experiment.py's fit_true_landscape_params — an
        # alpha=1e-6 GP (as used for loo_score/ls_score above) essentially
        # interpolates training points regardless of true noise level, so
        # in-sample residuals from THAT GP are uninformative (verified
        # empirically: an earlier version of this fix using in-sample
        # residuals from the interpolating GP collapsed to ~0.85 for every
        # objective, exactly reproducing the bug it was meant to fix). The
        # WhiteKernel GP is fit separately and only used for this component.
        from sklearn.gaussian_process import GaussianProcessRegressor as _GPR
        from sklearn.gaussian_process.kernels import Matern as _Matern, WhiteKernel as _WhiteKernel
        # NOTE ON MEASUREMENT-NOISE DETECTION (removed after verification):
        # An earlier version of this function added a WhiteKernel-based
        # noise_score component, intended to let C2 differentiate objectives
        # by their TRUE measurement noise (e.g. Waibel's kD CI/range=0.096
        # vs Tm's 0.013), independent of landscape roughness. This was
        # verified NOT to work: tested with ZERO injected noise across all
        # three objectives, the WhiteKernel still attributed substantial
        # "noise" to Tm (0.17-0.62) and near-zero to viscosity, at n=100
        # (obs_per_dim=6.25, well past any sparsity threshold). The
        # WhiteKernel term is absorbing landscape-fitting difficulty
        # (Tm's dose-response surface is a sum of two modest Gaussian
        # bumps, harder to reconstruct from sparse 16D categorical samples
        # than kD's single dominant mechanism), not measurement noise —
        # i.e. it duplicates what ls_score already measures, under a
        # different and misleading name, and cannot be fixed by adjusting
        # its activation threshold (a soft ramp toward an inverted signal
        # is still inverted, just smoother).
        #
        # Genuine per-objective measurement-noise detection (true C2, as
        # validated against Waibel's real CI-based noise ranking) requires
        # either (a) replicate measurements at the same input point, from
        # which noise can be estimated directly without confounding it with
        # landscape shape — which is how Waibel's real kD CIs are actually
        # computed, via linear-regression standard error across repeated
        # diffusion measurements, not GP fitting — or (b) an oracle that
        # simulates such replicates. Neither is implemented in this
        # synthetic oracle yet. Rather than ship a component that silently
        # measures the wrong thing, it is left out: trust is computed from
        # LOO calibration, ls/nn ratio, and uncertainty reduction only, and
        # this diagnostic should be understood as NOT currently validating
        # per-objective noise differentiation — only landscape trust.
        noise_score = 0.5  # neutral placeholder; component intentionally disabled

        if obs_per_dim < 2.0:
            trust = 0.6 * loo_score + 0.15 * ls_score + 0.25 * unc_score
        else:
            trust = 0.5 * loo_score + 0.3 * ls_score + 0.2 * unc_score
        scores[name] = float(np.clip(trust, 0, 1))

    return scores


def mixing_weight(trust_scores: dict, sparse: bool = False) -> float:
    """
    min-across-objectives trust -> LLM/GP mixing weight, per design decision.

    sparse=True inverts the mapping: at low obs_per_dim, "trust=0.8" is
    largely an artifact of loo_score's mean(|z|) statistic being fragile at
    small n (verified: a single catastrophic LOO miss, e.g. z=-11.72 on one
    left-out point, can be diluted by 9 more moderate residuals to a
    mean|z| that still lands inside the "calibrated" band) rather than
    genuine evidence the GP has learned real structure. Treating that as
    "lean on GP" is backwards in this regime: GP-sourced candidates come
    from strategy_mo_egbo, and mo_egbo is independently confirmed (via its
    own HV vs mo_random comparison, unrelated to this trust score) to
    underperform random search at n<~30 in this 16D space even after fixing
    its selection-order bias. So a high trust value here should map to
    LEANING ON THE LLM, not the GP, until obs_per_dim clears the threshold
    where ls_score/unc_score can actually differentiate (see
    per_objective_trust's own sparse-regime guards for that threshold).
    """
    w = min(trust_scores.values())
    return (1.0 - w) if sparse else w


# ── Pareto bookkeeping (dominance + duplicate filtering, no acquisition) ──────

def pareto_filter_candidates(candidates_n: np.ndarray, X_obs_n: np.ndarray,
                             oracle_predict_fn, dup_thresh: float = 0.05):
    """
    Filter candidates that are (a) near-duplicates of existing observations,
    or (b) dominated in the GP-predicted objective space by an existing
    observation's GP-predicted values. This is the "Pareto-bookkeeping and
    duplicate-filtering" role the GP layer plays when trust is low — NOT an
    acquisition function, just a filter to avoid wasting real experiments.
    """
    keep = []
    for i, c in enumerate(candidates_n):
        dists = np.linalg.norm(X_obs_n - c, axis=1)
        if dists.min() < dup_thresh:
            continue  # near-duplicate of an existing observation
        keep.append(i)
    return np.array(keep, dtype=int)


def pareto_front_of(Y: np.ndarray) -> np.ndarray:
    """Vectorised non-dominated sort, same logic as DiscreteMOExcipientOracle.pareto_front."""
    Yc = Y.copy()
    for j, d in enumerate(OBJECTIVE_DIRECTIONS):
        if d == "min":
            Yc[:, j] = -Yc[:, j]
    ge = (Yc[None, :, :] >= Yc[:, None, :]).all(axis=2)
    gt = (Yc[None, :, :] >  Yc[:, None, :]).any(axis=2)
    dominated = (ge & gt).any(axis=1)
    return np.where(~dominated)[0]


# ── Candidate pool generation (evolutionary-style, reused pattern) ────────────

def _make_pool(X_obs_n, Y_obs, bounds, rng, n=60):
    """Perturbation-around-Pareto-front + random exploration pool, in normalised space."""
    d = bounds.shape[0]
    pf_idx = pareto_front_of(Y_obs)
    seed_pool = X_obs_n[pf_idx] if len(pf_idx) > 0 else X_obs_n
    n_exploit = n // 2
    exploit = []
    while len(exploit) < n_exploit:
        base = seed_pool[rng.integers(len(seed_pool))]
        exploit.append(np.clip(base + rng.normal(0, 0.12, d), 0, 1))
    explore = rng.random((n - n_exploit, d))
    return np.vstack([exploit, explore])


# ── Strategies ──────────────────────────────────────────────────────────────────

def strategy_mo_random(oracle, X_obs, Y_obs, bounds, batch_size, rng, **kw):
    d = bounds.shape[0]
    cands = rng.random((batch_size * 6, d)) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
    return cands, {}


def strategy_mo_egbo(oracle, X_obs, Y_obs, bounds, batch_size, rng, **kw):
    """
    Multi-objective EGBO analogue: per-objective GP predictions + Pareto
    dominance ranking of predicted objective vectors, no scalarisation.
    """
    lo, hi = bounds[:,0], bounds[:,1]
    X_obs_n = (X_obs - lo) / (hi - lo + 1e-12)
    pool_n = _make_pool(X_obs_n, Y_obs, bounds, rng, n=60)
    pool_raw = pool_n * (hi - lo) + lo

    # Predict all 3 objectives at every candidate
    preds = np.zeros((len(pool_raw), 3))
    for j in range(3):
        gp, sc = _fit_gp_1d(X_obs, Y_obs[:, j])
        mu, sigma = gp.predict(sc.transform(pool_raw), return_std=True)
        beta = 2.0
        preds[:, j] = mu + beta * sigma * (1 if OBJECTIVE_DIRECTIONS[j]=="max" else -1)

    # Rank candidates by their own non-dominance within the predicted pool.
    pf_local = pareto_front_of(preds)
    # NOTE: pareto_front_of returns indices in ASCENDING array-position
    # order (an artifact of how np.where(~dominated) works), not in any
    # quality or diversity order. The pool itself is constructed by
    # _make_pool as [exploit candidates (first half) | explore candidates
    # (second half)] — verified empirically that in a typical n=10 batch,
    # ALL non-dominated candidates came from the exploit half and ZERO from
    # the explore half, and taking pf_local[:batch_size] then selected
    # exclusively from the exploit region purely because of array position,
    # not because exploit candidates were actually better. This starves
    # EGBO of exploration and is a plausible contributor to its large
    # variance and decelerating HV gains in real campaign runs. Fix:
    # shuffle before truncating, so selection isn't silently biased toward
    # whichever half of the pool happens to occupy lower indices.
    if len(pf_local) >= batch_size:
        shuffled = pf_local.copy()
        rng.shuffle(shuffled)
        selected = shuffled[:batch_size]
    else:
        # pad with next-best by crowding distance proxy (distance to Pareto set)
        remaining = [i for i in range(len(pool_raw)) if i not in pf_local]
        selected = list(pf_local)
        rng.shuffle(remaining)
        selected += remaining[:batch_size - len(selected)]
    return pool_raw[selected], {"pareto_local_size": len(pf_local)}


def strategy_mo_egbo_real(oracle, X_obs, Y_obs, bounds, batch_size, rng, **kw):
    """
    Genuine multi-objective EGBO: BoTorch qLogNoisyExpectedHypervolumeImprovement
    + pymoo UNSGA3(n_obj=3), replacing the lightweight Gaussian-perturbation
    stand-in (strategy_mo_egbo) with the actual evolutionary algorithm this
    project's single-objective EGBO work was validated with. Ported from
    egbo_mo.py's egbo_mo_test(), which was independently run and verified
    (monotonic HV growth, plausible Pareto front size on a synthetic
    3-objective problem) before this extraction — see egbo_mo.py's module
    docstring for the full verification note and why the run-before-trust
    discipline mattered here (BoTorch was not available to test this code
    at write time, only after).

    Requires botorch + pymoo. Falls back to strategy_mo_egbo (lightweight)
    on any import or runtime failure, with the failure logged rather than
    silently swallowed, so a real-EGBO run that silently degraded to the
    lightweight baseline is visible in the results rather than hidden.
    """
    try:
        import torch
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement)
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import SumMarginalLogLikelihood
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.transforms import normalize, unnormalize
        from botorch.optim.optimize import optimize_acqf
        from pymoo.algorithms.moo.unsga3 import UNSGA3
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.core.termination import NoTermination
        from pymoo.util.ref_dirs import get_reference_directions

        tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
        lo, hi = bounds[:, 0], bounds[:, 1]
        d = bounds.shape[0]
        M = Y_obs.shape[1]
        evo_candidates = 20

        # Convert to internal "all maximise" convention, matching
        # pareto_front_of elsewhere in this file, so ref_point/HV logic
        # stays consistent across the lightweight and real strategies.
        Y_int = Y_obs.copy()
        for j, direction in enumerate(OBJECTIVE_DIRECTIONS):
            if direction == "min":
                Y_int[:, j] = -Y_int[:, j]

        X_norm = (X_obs - lo) / (hi - lo + 1e-12)
        train_x = torch.tensor(X_norm, **tkwargs)
        train_y = torch.tensor(Y_int, **tkwargs)
        standard_bounds = torch.zeros(2, d, **tkwargs)
        standard_bounds[1] = 1.0

        # Reference point: fixed, same convention as run_mo_campaign's own
        # fixed_ref (anchored to the full oracle pool, not the growing
        # observed set) — avoids the reference-point-drift issue already
        # fixed once in this file for the HV *reporting* metric; here it
        # additionally serves as qLogNEHVI's own reference point input.
        Y_all_int = oracle._Y_raw.copy()
        for j, direction in enumerate(OBJECTIVE_DIRECTIONS):
            if direction == "min":
                Y_all_int[:, j] = -Y_all_int[:, j]
        ref_point = torch.tensor(
            Y_all_int.min(axis=0) -
            0.1 * (Y_all_int.max(axis=0) - Y_all_int.min(axis=0) + 1e-9),
            **tkwargs)

        models = [SingleTaskGP(train_x, train_y[:, j:j+1],
                               outcome_transform=Standardize(m=1))
                 for j in range(M)]
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll, max_attempts=1)

        acq_fn = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=ref_point, X_baseline=train_x,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
            prune_baseline=True, cache_root=True,
        )

        try:
            qbo_x, _ = optimize_acqf(
                acq_fn, bounds=standard_bounds, q=batch_size,
                num_restarts=2, raw_samples=16,
                options={"maxiter": 20},
            )
        except Exception:
            qbo_x = torch.rand(batch_size, d, **tkwargs)

        pf_idx_np = pareto_front_of(Y_obs)  # uses this file's own convention
        seed_pool_idx = pf_idx_np if len(pf_idx_np) > 0 else np.arange(len(Y_obs))
        seed_x = train_x[seed_pool_idx].cpu().numpy()
        if seed_x.shape[0] < evo_candidates:
            pad = rng.random((evo_candidates - seed_x.shape[0], d))
            seed_x = np.vstack([seed_x, pad])
        seed_x = seed_x[:max(evo_candidates, 2)]

        try:
            ref_dirs = get_reference_directions("energy", M, evo_candidates,
                                                seed=int(rng.integers(1e6)))
            pop_size = max(len(ref_dirs), evo_candidates, 2)
            algo = UNSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=seed_x)
            pm = PymooProblem(n_var=d, n_obj=M, n_constr=0,
                              xl=np.zeros(d), xu=np.ones(d))
            algo.setup(pm, termination=NoTermination())
            pop = algo.ask()
            pop_size_actual = len(pop)
            f_idx = seed_pool_idx[:pop_size_actual] if \
                len(seed_pool_idx) >= pop_size_actual else seed_pool_idx
            f_vals = -Y_int[f_idx]  # pymoo minimises
            if len(f_vals) >= pop_size_actual:
                pop.set("F", f_vals[:pop_size_actual])
            else:
                pad = np.tile(f_vals[-1:], (pop_size_actual - len(f_vals), 1))
                pop.set("F", np.vstack([f_vals, pad]))
            algo.tell(infills=pop)
            ea_x = torch.tensor(algo.ask().get("X"), **tkwargs)
        except Exception:
            ea_x = torch.rand(evo_candidates, d, **tkwargs)

        candidates = torch.cat([qbo_x, ea_x], dim=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acq_vals = []
            for i in range(candidates.shape[0]):
                try:
                    v = float(acq_fn(candidates[i].unsqueeze(0)).item())
                except Exception:
                    v = float("-inf")
                acq_vals.append(v)

        top_idx_sel = np.argsort(acq_vals)[-batch_size:]
        new_x_norm = candidates[top_idx_sel]
        new_x_raw = (unnormalize(new_x_norm, torch.tensor(
            np.column_stack([lo, hi]).T, **tkwargs)).cpu().numpy())
        return new_x_raw, {"real_egbo": True, "acq_vals": [acq_vals[i]
                           for i in top_idx_sel]}

    except Exception as e:
        warnings.warn(f"strategy_mo_egbo_real failed ({e}), "
                      f"falling back to lightweight strategy_mo_egbo — "
                      f"THIS RUN'S mo_egbo_real RESULTS ARE ACTUALLY THE "
                      f"LIGHTWEIGHT BASELINE, check logs for this warning.")
        cands, extra = strategy_mo_egbo(oracle, X_obs, Y_obs, bounds,
                                        batch_size, rng, **kw)
        extra["real_egbo"] = False
        extra["fallback_reason"] = str(e)
        return cands, extra


def strategy_mo_llm(oracle, X_obs, Y_obs, bounds, batch_size, rng,
                    prior_text="", model="qwen2.5:72b-instruct",
                    mock_llm=False, dataset="", **kw):
    """
    LLM proposes named formulations; per-objective trust score sets the
    LLM/GP mixing weight for the final candidate pool composition.
    """
    lo, hi = bounds[:,0], bounds[:,1]
    trust = per_objective_trust(X_obs, Y_obs)
    is_sparse = (len(X_obs) / bounds.shape[0]) < 2.0
    weight = mixing_weight(trust, sparse=is_sparse)
    # weight in [0,1] — HIGH trust => lean on GP more, UNLESS sparse regime,
    # in which case the mapping is inverted (see mixing_weight docstring).

    n_llm_propose = max(batch_size, int(batch_size * (1.5 - weight)))
    cache_key = f"{dataset}_{_obs_hash(X_obs, Y_obs)}"
    llm_proposals, cache_hit = llm_propose_mo_formulations(
        X_obs, Y_obs, prior_text, n_llm_propose, model, mock_llm, rng,
        cache_key=cache_key)

    # Diversity filter: remove near-duplicate LLM proposals BEFORE they
    # enter the pool. Verified this is a real problem, not a hypothesis:
    # sampled mock-LLM batches showed 15-25% of all pairwise distances
    # between proposals within a single batch were below 0.1 in the 16D
    # feature space, including exact duplicates (min distance = 0.000 in
    # 2/3 sampled seeds) — the LLM's perturb-around-Pareto-front logic
    # produces identical categorical choices (same amino acid/sugar/
    # surfactant) with only small concentration differences often enough
    # that several proposals per batch land at effectively the same point.
    # This is a plausible direct contributor to mo_llm's persistently large
    # but shallow Pareto front (hv_per_point ~600 vs ~1100-1400 for
    # mo_egbo/mo_random, unchanged across every other fix applied this
    # session) — a batch that nominally proposes 5-7 formulations but only
    # spans 3-4 truly distinct points isn't exploring as much as its
    # proposal count suggests.
    #
    # EXTENDED to also check against X_obs (all prior observations across
    # every previous batch), not just proposals within the current batch.
    # This was verified as a real, separate gap: sampled mock-LLM batches
    # showed 2-3 of 5 newly "proposed" formulations were EXACT duplicates
    # (distance=0.000) of formulations already tested in earlier batches —
    # the within-batch-only filter cannot catch this because it never looks
    # at X_obs at all. Re-testing an already-known formulation wastes a
    # real experiment slot regardless of whether the campaign intends to
    # explore or exploit at that point — this is not a diversity/
    # exploitation tradeoff, it is pure waste, so it is checked
    # unconditionally (hard threshold) rather than folded into the softer
    # within-batch spacing check. A tighter threshold (0.05) is used here
    # than the within-batch check (0.1) since this is specifically
    # catching near-exact repeats, not general clustering.
    #
    # NOTE: this does not address the softer case where the LLM proposes
    # formulations that are genuinely NEW input points but converge to
    # similar OBJECTIVE-SPACE outcomes (similar Tm/kD/viscosity via
    # different excipient combinations) — that is a legitimate
    # exploit-vs-explore tradeoff, not waste, and intentionally left
    # alone here rather than filtered, since the LLM should be allowed to
    # cluster in objective space when the evidence genuinely supports one
    # region being best. A proper fix for balancing that tradeoff would
    # score proposals by a combination of predicted quality (e.g. GP/
    # Pareto-dominance likelihood) and novelty rather than filtering by
    # input-space distance alone — flagged as a follow-up, not implemented
    # here to avoid stacking another unverified mechanism on top of this
    # already-compound set of fixes.
    if len(X_obs) > 0 and llm_proposals:
        X_obs_n_check = (X_obs - lo) / (hi - lo + 1e-12)
        not_repeat = []
        for p in llm_proposals:
            vec = formulation_to_vector(p["formulation"])
            if np.min(np.linalg.norm(X_obs_n_check - vec, axis=1)) > 0.05:
                not_repeat.append(p)
        llm_proposals = not_repeat

    if len(llm_proposals) > 1:
        diverse_proposals = [llm_proposals[0]]
        for p in llm_proposals[1:]:
            vec = formulation_to_vector(p["formulation"])
            is_diverse = all(
                np.linalg.norm(vec - formulation_to_vector(d["formulation"])) > 0.1
                for d in diverse_proposals
            )
            if is_diverse:
                diverse_proposals.append(p)
        llm_proposals = diverse_proposals

    llm_raw = (np.array([formulation_to_vector(p["formulation"])
                         for p in llm_proposals]) * (hi-lo) + lo
               if llm_proposals else np.zeros((0, len(lo))))

    # GP/evolutionary candidates fill the remainder, weighted by trust
    n_gp_needed = max(0, batch_size - int(round(batch_size * (1 - weight))))
    if len(llm_raw) > 0 and weight < 0.9:
        # low trust: mostly LLM, GP only for Pareto-bookkeeping/dedup
        gp_cands, _ = strategy_mo_egbo(oracle, X_obs, Y_obs, bounds, batch_size, rng)
        pool_raw = np.vstack([llm_raw, gp_cands]) if len(llm_raw) else gp_cands
    else:
        if len(llm_raw) >= batch_size:
            pool_raw = llm_raw
        else:
            gp_cands, _ = strategy_mo_egbo(oracle, X_obs, Y_obs, bounds, batch_size, rng)
            pool_raw = np.vstack([llm_raw, gp_cands]) if len(llm_raw) else gp_cands

    # Track which pool entries are LLM- vs GP-sourced BEFORE filtering, so
    # that after filtering removes some duplicates we still know the true
    # post-filter composition rather than assuming the pre-filter counts —
    # duplicate filtering can remove LLM- and GP-sourced candidates
    # unevenly (LLM proposals near an already-good point are more likely
    # to collide with an existing observation), so recomputing this after
    # filtering is required for the trust-weighting below to be correct.
    n_llm_pre_filter = len(llm_raw)
    source_is_llm = np.zeros(len(pool_raw), dtype=bool)
    source_is_llm[:n_llm_pre_filter] = True

    # Duplicate filtering (Pareto-bookkeeping role, not acquisition)
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
        # Trust-weighted selection probability: weight = mixing_weight means
        # "lean on GP by `weight`" (high trust -> more GP), so GP candidates
        # get probability mass `weight`, LLM candidates get `1-weight` —
        # split evenly WITHIN each source. This was previously uniform
        # random over the whole pool regardless of `weight`, which made the
        # mixing weight almost cosmetic (verified: weight=0.8 vs weight=0.3
        # produced pool compositions differing by only ~1 candidate slot,
        # and even that weak difference was then discarded by uniform
        # selection). Now the mixing weight directly sets selection
        # probability mass, not just how many candidates are OFFERED.
        gp_prob_each = weight / n_gp_in_pool
        llm_prob_each = (1.0 - weight) / n_llm_in_pool
        probs = np.where(source_is_llm, llm_prob_each, gp_prob_each)
        probs = probs / probs.sum()
        selected_idx = rng.choice(len(pool_raw), n_select, replace=False, p=probs)
    else:
        selected_idx = rng.choice(len(pool_raw), n_select, replace=False) \
                       if len(pool_raw) > n_select else np.arange(len(pool_raw))

    log_extra = {"trust": trust, "mixing_weight": weight, "cache_hit": cache_hit,
                "n_llm_proposed": len(llm_proposals),
                "llm_tags": [{"targets": p["targets"], "tradeoff": p["tradeoff"]}
                            for p in llm_proposals]}
    return pool_raw[selected_idx], log_extra


# ── Campaign runner ──────────────────────────────────────────────────────────────

def snap_query_mo(cands_raw, disc_oracle: DiscreteMOExcipientOracle):
    new_x, new_y = [], []
    for cand in cands_raw:
        y, idx = disc_oracle.query_mo(cand)
        new_x.append(disc_oracle._X_raw[idx])
        new_y.append(y)
    return new_x, new_y


def run_mo_campaign(disc_oracle, X_init, Y_init, budget, strategy_fn,
                    strategy_kwargs, batch_size=5, seed=0):
    bounds = disc_oracle.bounds()
    rng = np.random.default_rng(seed)
    X_obs, Y_obs = X_init.copy(), Y_init.copy()

    # CRITICAL FIX: reset _queried at the start of every seed's campaign,
    # and register the n_init points as already-queried.
    #
    # Bug found via direct verification: disc_oracle is built ONCE in
    # main() and the SAME instance is reused across all n_repeats seeds
    # (see the single `disc = oracle.make_discrete_oracle(...)` call
    # before the seed loop). Since `_queried` is mutable state on that
    # one shared instance and query_mo() never resets it, every seed
    # after the first inherited the full set of points queried by every
    # PRIOR seed's entire campaign — the pool of points a seed could
    # still select from shrank monotonically across the whole 15-seed
    # run, contaminating later seeds with earlier seeds' history.
    #
    # Separately, and independent of the cross-seed contamination: the
    # n_init points themselves were NEVER added to _queried at all (they
    # are read directly from disc_oracle._X_raw by index in
    # make_shared_inits, which has no knowledge of _queried). This meant
    # a seed's very first LLM or GP proposal could snap right back onto
    # one of its own n_init points — verified directly: batch-0 proposals
    # showed distance=0.000 to a prior X_obs row in a fresh 10-point-init
    # campaign, before any real campaign progress had a chance to cause
    # genuine reconvergence.
    disc_oracle._queried = set()
    X_all_s = disc_oracle._scaler.transform(disc_oracle._X_raw)
    X_init_s = disc_oracle._scaler.transform(X_init)
    for row in X_init_s:
        idx = int(np.argmin(np.linalg.norm(X_all_s - row, axis=1)))
        disc_oracle._queried.add(idx)

    # Fixed HV reference point, computed ONCE from the full oracle pool
    # (not the growing observed set). Recomputing the reference point every
    # batch from Y_obs.max() means the reference itself drifts upward as
    # better points are found — verified this made batch-to-batch HV values
    # incomparable (e.g. the kD component of the reference point shifted by
    # over 40% between an n=10 initial reference and the full-pool
    # reference). A rising HV trajectory under a drifting reference could
    # reflect genuine Pareto-front expansion, reference drift, or some mix
    # of both, with no way to tell which from the number alone. Anchoring
    # to the full pool's known range removes that ambiguity — every batch's
    # HV is measured against the same fixed aspiration point for the whole
    # campaign, so the trajectory genuinely reflects only front expansion.
    Y_all = disc_oracle._Y_raw.copy()
    Y_all_flipped = Y_all.copy()
    for j, d_ in enumerate(OBJECTIVE_DIRECTIONS):
        if d_ == "max":
            Y_all_flipped[:, j] = -Y_all_flipped[:, j]
    fixed_ref = (Y_all_flipped.max(axis=0) +
                0.1 * (Y_all_flipped.max(axis=0) - Y_all_flipped.min(axis=0) + 1e-9))

    hv_trajectory = []
    decisions = []
    n_batches = max(1, (budget - len(X_init)) // batch_size)

    for b in range(n_batches):
        step = len(X_init) + b * batch_size
        try:
            candidates, log_extra = strategy_fn(
                oracle=disc_oracle, X_obs=X_obs, Y_obs=Y_obs, bounds=bounds,
                batch_size=batch_size, rng=rng, **strategy_kwargs)
        except Exception as e:
            warnings.warn(f"Strategy failed batch {b}: {e}")
            d = bounds.shape[0]
            lo, hi = bounds[:,0], bounds[:,1]
            candidates = rng.uniform(lo, hi, (batch_size, d))
            log_extra = {"error": str(e)}

        new_x, new_y = snap_query_mo(candidates[:batch_size], disc_oracle)
        if new_x:
            X_obs = np.vstack([X_obs, new_x])
            Y_obs = np.vstack([Y_obs, new_y])

        pf_idx = pareto_front_of(Y_obs)
        try:
            from pymoo.indicators.hv import HV
            Y = Y_obs.copy()
            for j, d_ in enumerate(OBJECTIVE_DIRECTIONS):
                if d_ == "max":
                    Y[:, j] = -Y[:, j]
            Y_pf = Y[pf_idx]
            hv = float(HV(ref_point=fixed_ref)(Y_pf))
        except Exception:
            hv = float("nan")

        hv_trajectory.append(hv)
        decisions.append({"step": step, "n_obs": len(Y_obs),
                          "pareto_size": len(pf_idx), "hypervolume": hv,
                          **{k: v for k, v in log_extra.items()
                             if k != "llm_tags"},
                          "llm_tags": log_extra.get("llm_tags", [])})

    return {"hv_trajectory": hv_trajectory, "decisions": decisions,
            "X_obs": X_obs, "Y_obs": Y_obs}


# ── Shared initialisations ────────────────────────────────────────────────────

def make_shared_inits(disc_oracle, n_repeats, n_init, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    inits = []
    for _ in range(n_repeats):
        idx = rng.choice(len(disc_oracle), n_init, replace=False)
        inits.append((disc_oracle._X_raw[idx].copy(), disc_oracle._Y_raw[idx].copy()))
    return inits


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",    default="results_mo_campaign")
    parser.add_argument("--protein",    default="mAb_aggregation",
                        choices=["mAb_aggregation", "mAb_oxidation",
                                "enzyme_labile", "mixed"])
    parser.add_argument("--prior_level", default="L1", choices=["L1", "blank"])
    parser.add_argument("--n_repeats",  type=int, default=15)
    parser.add_argument("--budget",     type=int, default=40)
    parser.add_argument("--n_init",     type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--model",      default="qwen2.5:72b-instruct")
    parser.add_argument("--mock_llm",   action="store_true")
    parser.add_argument("--conditions", nargs="+",
                        default=["mo_egbo", "mo_llm", "mo_random"])
    parser.add_argument("--n_workers",  type=int, default=4)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_key = ("agg_L1" if args.prior_level == "L1" and "aggregation" in args.protein
                 else "oxid_L1" if args.prior_level == "L1" and "oxidation" in args.protein
                 else "blank")
    prior_text = MO_PRIORS.get(prior_key, MO_PRIORS["blank"])

    print(f"Protein: {args.protein}  prior_level: {args.prior_level}  "
          f"budget={args.budget}  n_init={args.n_init}  batch={args.batch_size}")

    oracle = MultiObjectiveExcipientOracle(
        protein=args.protein,
        tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10,  # Waibel-anchored
        seed=42,
    )
    disc = oracle.make_discrete_oracle(n_samples=500, seed=42)
    shared_inits = make_shared_inits(disc, args.n_repeats, args.n_init, rng_seed=42)

    STRATEGIES = {
        "mo_egbo":      (strategy_mo_egbo,      {}),
        "mo_egbo_real": (strategy_mo_egbo_real, {}),
        "mo_random":    (strategy_mo_random,    {}),
        "mo_llm":       (strategy_mo_llm,       {"prior_text": prior_text,
                                                  "model": args.model,
                                                  "mock_llm": args.mock_llm,
                                                  "dataset": args.protein}),
    }

    all_rows = []
    for cond in args.conditions:
        fn, kwargs = STRATEGIES[cond]
        print(f"  {cond}...", end="", flush=True)
        for seed_idx, (X_init, Y_init) in enumerate(shared_inits):
            res = run_mo_campaign(disc, X_init, Y_init, args.budget, fn, kwargs,
                                  batch_size=args.batch_size, seed=seed_idx)
            final_hv = res["hv_trajectory"][-1] if res["hv_trajectory"] else float("nan")
            final_pf_size = len(pareto_front_of(res["Y_obs"]))
            all_rows.append({"protein": args.protein, "prior_level": args.prior_level,
                             "condition": cond, "seed": seed_idx,
                             "final_hv": final_hv, "final_pf_size": final_pf_size,
                             "n_obs": len(res["Y_obs"])})

            cond_dir = out_dir / args.protein / cond
            cond_dir.mkdir(parents=True, exist_ok=True)
            with open(cond_dir / f"seed_{seed_idx:03d}.json", "w") as f:
                json.dump({"seed": seed_idx, "hv_trajectory": res["hv_trajectory"],
                          "decisions": res["decisions"]}, f, indent=2)
            print(".", end="", flush=True)
        print()

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "mo_campaign_summary.csv", index=False)
    print(f"\nSaved: {out_dir / 'mo_campaign_summary.csv'}")

    print(f"\n{'='*60}\nRESULTS ({args.protein}, prior={args.prior_level})\n{'='*60}")
    agg = df.groupby("condition").agg(
        hv_mean=("final_hv","mean"), hv_std=("final_hv","std"),
        pf_mean=("final_pf_size","mean"),
    ).sort_values("hv_mean", ascending=False)
    for cond, row in agg.iterrows():
        print(f"  {cond:<12} HV={row.hv_mean:.1f}±{row.hv_std:.1f}  "
              f"Pareto_size={row.pf_mean:.1f}")


if __name__ == "__main__":
    main()
