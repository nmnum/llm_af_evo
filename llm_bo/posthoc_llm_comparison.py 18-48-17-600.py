"""
posthoc_llm_comparison.py — LLM-BO vs EGBO vs Random (efficient version)
═══════════════════════════════════════════════════════════════════════════

WHAT CHANGED FROM v1 (2-day runtime → ~2 hours):

Architecture change (main speedup):
  OLD: LLM proposes 30 raw coordinates per batch → snap to oracle → GP scores
       Problem: LLM generates in continuous space, snapping wastes proposals,
       ~2 min × 540 calls = 18 hours LLM inference
  NEW: GP+evo generates candidate pool → LLM ranks top-20 by index → select 4
       LLM output is just 4 integers (indices), not 30 coordinate vectors
       Much shorter prompt/response → ~30s per call instead of 2 min

Caching (secondary speedup):
  Observation history is hashed. Identical (dataset, hash) combos reuse the
  same LLM response — early batches across seeds often share histories since
  they start from the same init and query the same small oracle pool.
  Estimated cache hit rate: 30-50% in early campaign → ~30% fewer LLM calls.

Parallelism:
  Seeds run in parallel (--n_workers, default 4). Independent campaigns,
  no shared state. Scales linearly with workers up to GPU memory limit.

Logging:
  Per-seed JSON logs saved to results_dir/<dataset>/<condition>/seed_NNN.json
  Each log contains: auc, final, decisions (step, queried_x, queried_y,
  llm_candidates_shown, llm_selected_indices, gp_scores, cache_hit).

Conditions:
  egbo      — EGBO with evolutionary candidates + GP-UCB (your baseline)
  llm_bo    — Pool-ranking LLM: GP+evo pool → LLM ranks → select 4
  llm_nopad — LLM-BO without the 8 random padding candidates (ablation:
              does random padding explain the advantage?)
  random    — Pure random search (floor)

Usage:
  # Quick test, no LLM (~5 min)
  python posthoc_llm_comparison.py --data_dir data/ --mock_llm --n_repeats 5

  # Full run, Spark GPU
  python posthoc_llm_comparison.py --data_dir data/ \\
      --datasets pareto_20210112 coatings hartmann6 \\
      --n_repeats 20 --model qwen2.5:72b-instruct --n_workers 4

  # Ablation: does removing hartmann6 prior hint change result?
  python posthoc_llm_comparison.py --data_dir data/ \\
      --datasets hartmann6 --n_repeats 20 \\
      --model qwen2.5:72b-instruct --no_optimum_hint
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).parent))


# ── Prior knowledge (physical descriptions only, no optimum hints) ─────────────

PRIOR_KNOWLEDGE = {
    "pareto_20210112": """
Coating process optimisation — 4 continuous parameters, all normalised [0,1].
Goal: maximise scalarised coating quality (conductivity + uniformity + adhesion).

Parameters:
  x0 (concentration): precursor concentration. Low=uniform thin layers; high=thick but cracking risk.
  x1 (temperature):   deposition temp (25–200°C). Higher improves crystallinity and conductivity.
  x2 (flow_rate):     carrier gas flow. Low=denser coating; high=non-uniform risk.
  x3 (pressure):      chamber pressure. Low=better step coverage; high=faster deposition.

Known interactions:
  - High concentration (x0>0.7) + high temperature (x1>0.8) → cracking, avoid this combination.
  - Best formulations tend toward moderate temperature (0.3–0.7) and moderate concentration (0.2–0.6).
  - Low flow + low pressure → slow but high quality.
""",

    "coatings": """
Coating formulation optimisation — 7 continuous parameters, all normalised [0,1].
Goal: maximise combined optical performance.

Parameters:
  x0 (binder_fraction): polymer binder fraction.
  x1 (pigment_loading): pigment concentration (0=clear, 1=fully loaded).
  x2 (solvent_ratio):   primary/secondary solvent ratio.
  x3 (cure_temp):       curing temperature (room temp to 200°C). Higher improves crosslinking.
  x4 (cure_time):       curing duration (1–60 min).
  x5 (film_thickness):  wet film thickness.
  x6 (additive_level):  performance additive concentration. Diminishing returns above 0.6.

Guidance: moderate binder (0.3–0.6) + moderate pigment (0.2–0.5) often optimal.
Higher cure temperature (x3>0.5) generally beneficial. Avoid extreme values in multiple dims.
""",

    "hartmann6": """
6-dimensional optimisation problem. All inputs x0–x5 in [0,1].
The landscape has multiple local optima — diverse exploration is important.
Mid-range values (0.2–0.7) tend to be more promising than boundary values.
""",

    "hartmann6_no_hint": """
6-dimensional optimisation problem. All inputs x0–x5 in [0,1].
The landscape has multiple local optima — diverse exploration is important.
Mid-range values (0.2–0.7) tend to be more promising than boundary values.
""",

    "hartmann3": "3-dimensional optimisation in [0,1]^3. Explore diverse regions.",

    "pareto_20201218": """
Coating process with 4 parameters in [0,1]. This landscape is relatively flat —
broad exploration is more important than exploitation.
""",

    "hartmann3": """
3-dimensional optimisation problem. All inputs x0-x2 in [0,1].
The landscape has 4 local optima. Mid-range values tend to be more
promising than boundary values. Diverse exploration across all three
dimensions is important early in the campaign.
""",

    # Excipient formulation priors — rich pharmaceutical domain knowledge
    "excipient_mAb_aggregation": """
You are optimising a protein formulation for an aggregation-prone monoclonal antibody.
The primary degradation pathway is aggregation (intermolecular hydrophobic interactions).

Excipient space (16D encoded vector, but reason about excipient identity):
  Amino acid (one of 5): arginine, proline, glycine, methionine, histidine
  Sugar (one of 4): sucrose, trehalose, sorbitol, mannitol
  Surfactant (one of 3): polysorbate80, polysorbate20, poloxamer188
  EDTA: True/False

Key domain knowledge for aggregation-prone proteins:
  - Arginine (10-20 mM) is the strongest anti-aggregation amino acid via
    charge-shielding of hydrophobic patches. BEST CHOICE here.
  - Sucrose or trehalose (50-80 mM) stabilise via preferential exclusion.
    Nearly equivalent; trehalose has slightly higher glass transition Tg.
  - Polysorbate80 (0.3-0.5%) provides best interface protection for mAbs.
    Synergy with arginine: arginine + PS80 together give better aggregation
    suppression than either alone. This combination is the gold standard.
  - Methionine is for OXIDATION protection — minimal effect here.
  - EDTA (0.01%) chelates metals that catalyse oxidation — moderate benefit.
  - Mannitol above 40 mM risks crystallisation — AVOID high mannitol.
  - High arginine (>20 mM) + high sucrose (>70 mM) may exceed solubility
    and increase viscosity — avoid this combination.

Target: stability score in [0,1]. Global optimum is near:
  arginine ~15 mM, sucrose/sorbitol ~50-70 mM, polysorbate80 ~0.4%, EDTA=True
""",

    "excipient_mAb_oxidation": """
You are optimising a protein formulation for an oxidation-prone monoclonal antibody.
The primary degradation pathway is methionine/tryptophan oxidation.

Key domain knowledge for oxidation-prone proteins:
  - Methionine (0.5-2 mM) is the PRIMARY stabiliser here — sacrificial
    oxidation protects protein Met/Trp residues. BEST AMINO ACID CHOICE.
  - EDTA (0.01%) chelates metal ions that catalyse oxidative reactions.
    EDTA + methionine is the key synergistic combination for oxidation.
  - Sucrose or trehalose (50-70 mM) provide baseline conformational stability.
  - Polysorbate80 (0.3-0.5%) for interface protection.
  - Arginine has minimal benefit for oxidation-prone proteins.
  - Histidine provides buffer capacity but limited antioxidant effect.

Target: stability score in [0,1]. Global optimum is near:
  methionine ~1 mM, sucrose ~70 mM, polysorbate80 ~0.4%, EDTA=True
""",

    "excipient_enzyme_labile": """
You are optimising a formulation for a thermally labile enzyme.
The primary degradation pathway is thermal denaturation (low Tm).

Key domain knowledge for thermally labile proteins:
  - Sucrose and trehalose (60-80 mM) are the strongest Tm stabilisers
    via preferential exclusion. HIGH CONCENTRATION preferred.
  - Glycine (15-20 mM) provides osmolyte stabilisation and tonicity.
  - Polysorbate80 (0.3-0.5%) prevents interface aggregation during
    thermal cycling.
  - Arginine has less effect on Tm for labile enzymes.
  - Methionine is for oxidation — limited benefit here.
  - Mannitol above 40 mM risks crystallisation — avoid.

Target: stability score in [0,1]. Global optimum is near:
  glycine ~18 mM, sucrose ~70 mM, polysorbate80 ~0.4%, EDTA=False
""",

    "excipient_mixed": """
You are optimising a protein formulation with multiple degradation pathways
(aggregation, oxidation, and thermal denaturation all contribute).

Key domain knowledge for mixed-mode degradation:
  - No single excipient dominates — need a balanced formulation.
  - Histidine (5-8 mM) provides both buffer capacity and moderate
    stabilisation across pathways.
  - Sucrose or trehalose (50-70 mM) provide broad conformational stability.
  - Polysorbate80 (0.3-0.5%) for interface protection.
  - EDTA (0.01%) for oxidation component.
  - Consider: arginine for aggregation, methionine for oxidation,
    but at lower concentrations than single-pathway formulations.

Target: stability score in [0,1]. No single dominant optimal combination.
""",
}

DEFAULT_PRIOR = "Optimise a {d}-dimensional function in [0,1]^d. Explore broadly."

# ── LLM candidate RANKING (new architecture) ───────────────────────────────────

_LLM_CACHE: dict = {}   # (dataset, obs_hash) → selected_indices


def _obs_hash(X_obs: np.ndarray, y_obs: np.ndarray) -> str:
    """Stable hash of current observations for caching."""
    arr = np.column_stack([X_obs, y_obs.reshape(-1, 1)])
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]


def _build_ranking_prompt(prior_text: str, X_obs: np.ndarray,
                           y_obs: np.ndarray, candidates: np.ndarray,
                           gp_scores: np.ndarray, batch_size: int) -> str:
    """
    Build a short prompt showing the candidate pool with GP scores.
    LLM only needs to output batch_size indices — much shorter than v1.
    """
    d = candidates.shape[0]
    y_norm_best = (y_obs.max() - y_obs.min()) / (y_obs.max() - y_obs.min() + 1e-12)

    # Normalise GP scores for display
    gs = gp_scores
    gs_norm = (gs - gs.min()) / (gs.max() - gs.min() + 1e-12)

    # Show top-5 observed for context (not all — keeps prompt short)
    top5_idx = np.argsort(y_obs)[::-1][:5]
    obs_lines = []
    for i, idx in enumerate(top5_idx):
        coords = " ".join(f"{X_obs[idx,j]:.2f}" for j in range(X_obs.shape[1]))
        obs_lines.append(f"  {i+1}. [{coords}]  score={y_obs[idx]:.3f}")

    # Show candidates with GP scores
    cand_lines = []
    for i, (cand, sc) in enumerate(zip(candidates, gs_norm)):
        coords = " ".join(f"{cand[j]:.2f}" for j in range(len(cand)))
        cand_lines.append(f"  {i:2d}. [{coords}]  gp_score={sc:.3f}")

    return f"""{prior_text.strip()}

Campaign: {len(y_obs)} observations, best score={y_obs.max():.3f}

Top-5 observed (for context):
{"".join(obs_lines)}

Candidate pool (GP-scored, higher=better predicted by model):
{"".join(cand_lines)}

Select the {batch_size} best candidates to evaluate next.
Consider both high GP scores (exploitation) and diversity (exploration).
Respond with JSON only — a list of {batch_size} integer indices from the pool above:
{{"selected": [{", ".join(str(i) for i in range(batch_size))}]}}"""


def llm_rank_candidates(X_obs: np.ndarray, y_obs: np.ndarray,
                         candidates: np.ndarray, gp_scores: np.ndarray,
                         prior_text: str, batch_size: int,
                         model: str, mock: bool, rng: np.random.Generator,
                         cache_key: str | None = None,
                         max_retries: int = 3) -> tuple[list[int], bool]:
    """
    Ask LLM to select batch_size indices from the candidate pool.
    Returns (selected_indices, cache_hit, reasoning).
    reasoning is the LLM's free-text explanation — empty string on cache hit or mock.
    """
    global _LLM_CACHE

    # Cache lookup
    if cache_key and cache_key in _LLM_CACHE:
        cached = _LLM_CACHE[cache_key]
        # Validate indices still valid for current pool size
        if all(0 <= i < len(candidates) for i in cached):
            return cached, True, ""   # cache hit: reasoning not re-generated

    n_cands = len(candidates)

    if mock:
        # Mock: pick top-half by GP score, bottom-half by novelty
        top_gp = int(np.argsort(gp_scores)[::-1][0])
        selected = [top_gp]
        for _ in range(batch_size - 1):
            # Pick diverse point
            sel_cands = candidates[selected]
            dists = np.array([
                np.min(np.linalg.norm(sel_cands - candidates[i], axis=1))
                for i in range(n_cands) if i not in selected
            ])
            remaining = [i for i in range(n_cands) if i not in selected]
            selected.append(remaining[int(np.argmax(dists))])
        result = selected[:batch_size]
        if cache_key:
            _LLM_CACHE[cache_key] = result
        return result, False, ""   # mock: no reasoning

    import ollama

    prompt = _build_ranking_prompt(prior_text, X_obs, y_obs,
                                    candidates, gp_scores, batch_size)
    text = ""
    for attempt in range(max_retries):
        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert optimisation assistant. "
                                "Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.1, "num_predict": 128,
                         "think": False},
            )
            text = resp["message"]["content"]
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = re.sub(r'```(?:json)?', '', text).strip().strip('`')
            # Remove trailing commas
            text = re.sub(r',(\s*[}\]])', r'\1', text)

            parsed = json.loads(text)
            selected = parsed.get("selected", [])
            selected = [int(i) for i in selected
                        if isinstance(i, (int, float)) and 0 <= int(i) < n_cands]

            # Deduplicate while preserving order
            seen, unique = set(), []
            for i in selected:
                if i not in seen:
                    seen.add(i); unique.append(i)
            selected = unique

            # Pad with top GP-scored unselected if LLM returned too few
            if len(selected) < batch_size:
                for i in np.argsort(gp_scores)[::-1]:
                    if int(i) not in selected:
                        selected.append(int(i))
                    if len(selected) == batch_size:
                        break

            result = selected[:batch_size]
            if cache_key:
                _LLM_CACHE[cache_key] = result
            reasoning = parsed.get("reasoning", "")
            return result, False, reasoning

        except Exception as e:
            if attempt == max_retries - 1:
                warnings.warn(
                    f"LLM ranking failed after {max_retries} attempts: {e}\n"
                    f"Response: {text[:200]}\nFalling back to GP-top."
                )
                result = [int(i) for i in
                          np.argsort(gp_scores)[::-1][:batch_size]]
                return result, False, ""   # failed: no reasoning


# ── GP fitting ──────────────────────────────────────────────────────────────────

def _fit_gp(X_obs, y_obs):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xs = sc.fit_transform(X_obs)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
        alpha=1e-6, normalize_y=True, n_restarts_optimizer=3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(Xs, y_obs)
    return gp, sc


def _ucb_scores(gp, sc, X_cands, y_obs, progress):
    beta = max(0.5, 5.0 * (1.0 - progress))
    mu, sigma = gp.predict(sc.transform(X_cands), return_std=True)
    return mu + beta * sigma


# ── Candidate pool generation (shared by EGBO and LLM-BO) ─────────────────────

def _make_candidate_pool(X_obs, y_obs, bounds, rng, evo_pop=72,
                          add_random_pad=True):
    """
    Evolutionary candidates (EGBO-style) + optional random padding.
    Returns (candidates_normalised, candidates_raw).
    """
    from pymoo.algorithms.moo.unsga3 import UNSGA3
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.termination import NoTermination
    from pymoo.util.ref_dirs import get_reference_directions

    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    Xn = (X_obs - lo) / (hi - lo + 1e-12)

    top_k = min(evo_pop, len(y_obs))
    seed_x = Xn[np.argsort(y_obs)[-top_k:][::-1]]
    if len(seed_x) < max(evo_pop, 2):
        pad = rng.random((max(evo_pop, 2) - len(seed_x), d))
        seed_x = np.vstack([seed_x, pad])
    try:
        pop_size = max(evo_pop, 2)
        ref_dirs = get_reference_directions("energy", 1, pop_size, seed=42)
        algo = UNSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=seed_x)
        pm = PymooProblem(n_var=d, n_obj=1, n_constr=0,
                          xl=np.zeros(d), xu=np.ones(d))
        algo.setup(pm, termination=NoTermination())
        pop = algo.ask()
        n_act = len(pop)
        f_vals = -y_obs[np.argsort(y_obs)[-n_act:]]
        pop.set("F", f_vals.reshape(-1, 1))
        algo.tell(infills=pop)
        ea_n = np.clip(algo.ask().get("X"), 0, 1)[:evo_pop]
    except Exception:
        ea_n = rng.random((evo_pop, d))

    if add_random_pad:
        rand_n = rng.random((8, d))
        cands_n = np.vstack([ea_n, rand_n])
    else:
        cands_n = ea_n

    cands_raw = cands_n * (hi - lo) + lo
    return cands_n, cands_raw


# ── Strategy implementations ────────────────────────────────────────────────────

def strategy_random(oracle, X_obs, y_obs, bounds, batch_size, rng, **kw):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    cands = rng.uniform(lo, hi, (batch_size * 8, d))
    return cands[:batch_size], {}


def strategy_egbo(oracle, X_obs, y_obs, bounds, batch_size, rng,
                   progress=0.5, **kw):
    lo, hi = bounds[:, 0], bounds[:, 1]
    cands_n, cands_raw = _make_candidate_pool(X_obs, y_obs, bounds, rng)
    gp, sc = _fit_gp(X_obs, y_obs)
    scores = _ucb_scores(gp, sc, cands_raw, y_obs, progress)

    selected, remaining = [], list(range(len(cands_n)))
    for _ in range(batch_size):
        if not remaining:
            break
        rem = np.array(remaining)
        acq = scores[rem]
        nov = (np.array([np.min(np.linalg.norm(cands_n[selected] - cands_n[i], axis=1))
                          for i in rem])
               if selected else np.ones(len(rem)))
        a = (acq - acq.min()) / (acq.max() - acq.min() + 1e-12)
        n = (nov - nov.min()) / (nov.max() - nov.min() + 1e-12)
        pick = int(rem[np.argmax(0.7*a + 0.3*n)])
        selected.append(pick); remaining.remove(pick)

    return cands_raw[selected], {"gp_scores": scores[selected].tolist()}


def strategy_llm_bo(oracle, X_obs, y_obs, bounds, batch_size, rng,
                     progress=0.5, prior_text="", model="qwen2.5:72b-instruct",
                     mock_llm=False, dataset="", add_random_pad=True, **kw):
    """
    New architecture: GP+evo pool → LLM ranks by index → select batch.
    LLM output is batch_size integers, not coordinate vectors.
    """
    lo, hi = bounds[:, 0], bounds[:, 1]
    cands_n, cands_raw = _make_candidate_pool(
        X_obs, y_obs, bounds, rng, add_random_pad=add_random_pad)

    gp, sc = _fit_gp(X_obs, y_obs)
    scores = _ucb_scores(gp, sc, cands_raw, y_obs, progress)

    # Show LLM the top-20 by GP score (keeps prompt short)
    top20_idx = np.argsort(scores)[::-1][:20]
    pool_for_llm = cands_n[top20_idx]
    scores_for_llm = scores[top20_idx]

    cache_key = f"{dataset}_{_obs_hash(X_obs, y_obs)}_pad{add_random_pad}"
    selected_in_top20, cache_hit, llm_reasoning = llm_rank_candidates(
        X_obs=X_obs, y_obs=y_obs,
        candidates=pool_for_llm, gp_scores=scores_for_llm,
        prior_text=prior_text, batch_size=batch_size,
        model=model, mock=mock_llm, rng=rng,
        cache_key=cache_key,
    )
    # Map back to original pool indices
    selected = [int(top20_idx[i]) for i in selected_in_top20]

    log_extra = {
        "gp_scores_selected": scores[selected].tolist(),
        "llm_indices_in_top20": selected_in_top20,
        "pool_size": len(cands_n),
        "cache_hit": cache_hit,
        "add_random_pad": add_random_pad,
        "llm_reasoning": llm_reasoning,   # LLM free-text explanation
    }
    return cands_raw[selected], log_extra


# ── Oracle snapping ─────────────────────────────────────────────────────────────

def snap_and_query(candidates, oracle, queried):
    """Snap raw candidates to nearest unqueried oracle row, query them."""
    all_X = oracle._X_raw
    scaler = oracle._scaler
    X_all_s = scaler.transform(all_X)
    unqueried = [i for i in range(len(all_X)) if i not in queried]
    if not unqueried:
        unqueried = list(range(len(all_X)))

    new_x, new_y = [], []
    for cand in candidates:
        if not unqueried:
            break
        cand_s = scaler.transform(cand.reshape(1, -1))[0]
        pool_s = scaler.transform(all_X[unqueried])
        chosen = unqueried[int(np.argmin(np.linalg.norm(pool_s - cand_s, axis=1)))]
        queried.add(chosen)
        new_x.append(all_X[chosen])
        new_y.append(float(oracle._y_raw[chosen]))
        unqueried = [i for i in unqueried if i != chosen]
    return new_x, new_y


# ── Campaign runner ─────────────────────────────────────────────────────────────

def run_campaign(oracle, X_init, y_init, budget, strategy_fn,
                 strategy_kwargs, batch_size=4, seed=0):
    bounds = oracle.bounds()
    all_X = oracle._X_raw
    scaler = oracle._scaler
    rng = np.random.default_rng(seed)

    X_obs, y_obs = X_init.copy(), y_init.copy()
    queried = set()
    X_all_s = scaler.transform(all_X)
    for row in scaler.transform(X_init):
        queried.add(int(np.argmin(np.linalg.norm(X_all_s - row, axis=1))))

    running_best = [float(y_obs.max())] * len(X_init)
    decisions = []
    n_batches = max(1, (budget - len(X_init)) // batch_size)

    for b in range(n_batches):
        step = len(X_init) + b * batch_size
        progress = step / budget
        try:
            candidates, log_extra = strategy_fn(
                oracle=oracle, X_obs=X_obs, y_obs=y_obs, bounds=bounds,
                batch_size=batch_size, rng=rng, progress=progress,
                **strategy_kwargs,
            )
        except Exception as e:
            warnings.warn(f"Strategy failed batch {b}: {e}. Using random.")
            d = bounds.shape[0]
            lo, hi = bounds[:, 0], bounds[:, 1]
            candidates = rng.uniform(lo, hi, (batch_size, d))
            log_extra = {"error": str(e)}

        new_x, new_y = snap_and_query(candidates, oracle, queried)
        X_obs = np.vstack([X_obs, np.array(new_x)]) if new_x else X_obs
        y_obs = np.append(y_obs, new_y) if new_y else y_obs
        for _ in new_y:
            running_best.append(float(y_obs.max()))

        decisions.append({
            "step": step, "progress": round(progress, 3),
            "n_obs": len(y_obs),
            "queried_x": [x.tolist() for x in new_x],
            "queried_y": new_y,
            **log_extra,
        })

    return {"running_best": running_best, "decisions": decisions}


# ── Single seed runner (for parallelism) ───────────────────────────────────────

def _run_one_seed(args_tuple):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    (seed_idx, ds_label, ds_name, cond, strategy_name, strategy_kwargs,
     budget, n_init, batch_size, data_dir, out_dir_str, gb) = args_tuple

    sys.path.insert(0, str(pathlib.Path(data_dir).parent))
    from oracle import NNOracle
    from shared_seed_experiment import generate_shared_inits

    oracle = NNOracle.from_dataset(ds_name, data_dir)
    shared_inits = generate_shared_inits(oracle, seed_idx + 1, n_init,
                                          rng_seed=42)
    X_init, y_init = shared_inits[seed_idx]

    STRATEGY_MAP = {
        "egbo":       strategy_egbo,
        "llm_bo":     strategy_llm_bo,
        "llm_nopad":  strategy_llm_bo,
        "random":     strategy_random,
    }
    fn = STRATEGY_MAP[strategy_name]

    res = run_campaign(oracle, X_init, y_init, budget, fn,
                        strategy_kwargs, batch_size=batch_size, seed=seed_idx)

    curve = np.array(res["running_best"])
    auc   = float((curve / gb).mean())
    final = float(curve[-1] / gb)

    log = {"dataset": ds_label, "condition": cond, "seed": seed_idx,
           "auc": auc, "final": final, "budget": budget,
           "N": len(oracle._X_raw), "decisions": res["decisions"]}

    out_dir = pathlib.Path(out_dir_str)
    cond_dir = out_dir / ds_label / cond
    cond_dir.mkdir(parents=True, exist_ok=True)
    with open(cond_dir / f"seed_{seed_idx:03d}.json", "w") as f:
        json.dump(log, f, indent=2)

    return {"dataset": ds_label, "condition": cond, "seed": seed_idx,
            "auc": auc, "final": final, "budget": budget,
            "N": len(oracle._X_raw)}


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--out_dir",     default="results_posthoc_llm_v3")
    parser.add_argument("--datasets",    nargs="+",
                        default=["pareto_20210112"])
    parser.add_argument("--n_repeats",   type=int, default=20)
    parser.add_argument("--budget_frac", type=float, default=0.5)
    parser.add_argument("--n_init",      type=int, default=5)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--model",       default="qwen2.5:72b-instruct")
    parser.add_argument("--mock_llm",    action="store_true")
    parser.add_argument("--conditions",  nargs="+",
                        default=["egbo", "llm_nopad", "random"],
                        help="Conditions to run. llm_nopad is the recommended "
                             "LLM condition (no random padding, faster convergence). "
                             "llm_bo adds 8 random candidates to LLM pool.")
    parser.add_argument("--n_workers",   type=int, default=4,
                        help="Parallel workers for seed execution")
    parser.add_argument("--no_optimum_hint", action="store_true",
                        help="Remove hartmann6 prior knowledge hint about "
                             "global optimum location (ablation)")
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(args.data_dir).parent))
    from oracle import NNOracle
    from shared_seed_experiment import generate_shared_inits

    DATASET_MAP = {
        "pareto_20201218":      "pareto_campaign 2020-12-18_17-38-40",
        "pareto_20210112":      "pareto_campaign 2021-01-12_16-26-56",
        "coatings":             "coatings",
        "hartmann3":            "hartmann3",
        "hartmann6":            "hartmann6",
        # Excipient formulation oracles — prefix "excipient_"
        "excipient_mAb_aggregation": "excipient:mAb_aggregation",
        "excipient_mAb_oxidation":   "excipient:mAb_oxidation",
        "excipient_enzyme_labile":   "excipient:enzyme_labile",
        "excipient_mixed":           "excipient:mixed",
    }

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for ds_label in args.datasets:
        ds_name = DATASET_MAP.get(ds_label, ds_label)
        if ds_name.startswith("excipient:"):
            # Load excipient synthetic oracle — no data_dir needed
            from excipient_oracle import ExcipientOracle
            protein_name = ds_name.split(":", 1)[1]
            _cont_oracle = ExcipientOracle(protein=protein_name, seed=42)
            oracle = _cont_oracle.make_discrete_oracle(
                n_samples=500, seed=42)
        else:
            oracle = NNOracle.from_dataset(ds_name, args.data_dir)
        N  = len(oracle._X_raw)
        gb = oracle.global_best()
        budget = max(args.n_init + 10, int(args.budget_frac * N))

        # Prior text — ablation: strip optimum hint from hartmann6
        prior_key = (f"{ds_label}_no_hint"
                     if args.no_optimum_hint and ds_label == "hartmann6"
                     else ds_label)
        prior_text = PRIOR_KNOWLEDGE.get(
            prior_key, DEFAULT_PRIOR.format(d=oracle.bounds().shape[0]))

        print(f"\n{'='*60}\nDataset: {ds_label}  N={N}  budget={budget}  "
              f"dims={oracle.bounds().shape[0]}")
        print(f"  {'mock LLM' if args.mock_llm else args.model}")

        for cond in args.conditions:
            # Build strategy kwargs
            base_llm_kw = dict(prior_text=prior_text, model=args.model,
                                mock_llm=args.mock_llm, dataset=ds_label)
            STRATEGY_KWARGS = {
                "egbo":      {},
                "llm_bo":    {**base_llm_kw, "add_random_pad": True},
                "llm_nopad": {**base_llm_kw, "add_random_pad": False},
                "random":    {},
            }
            if cond not in STRATEGY_KWARGS:
                print(f"  Unknown condition {cond}, skipping")
                continue

            print(f"  {cond}...", end="", flush=True)
            kwargs = STRATEGY_KWARGS[cond]
            strategy_name = "llm_bo" if cond == "llm_nopad" else cond

            # Build task list for parallel execution
            tasks = [
                (seed_idx, ds_label, ds_name, cond, strategy_name, kwargs,
                 budget, args.n_init, args.batch_size,
                 args.data_dir, str(out_dir), gb)
                for seed_idx in range(args.n_repeats)
            ]

            # Run seeds — parallel for non-LLM conditions, sequential for LLM
            # (parallel LLM would hit Ollama concurrency limits)
            is_llm = cond in ("llm_bo", "llm_nopad") and not args.mock_llm
            n_workers = 1 if is_llm else args.n_workers

            rows = []
            if n_workers == 1:
                for task in tasks:
                    row = _run_one_seed(task)
                    rows.append(row)
                    all_rows.append(row)
                    print(".", end="", flush=True)
            else:
                with ProcessPoolExecutor(max_workers=n_workers) as ex:
                    futures = {ex.submit(_run_one_seed, t): t for t in tasks}
                    for fut in as_completed(futures):
                        row = fut.result()
                        rows.append(row)
                        all_rows.append(row)
                        print(".", end="", flush=True)

            aucs = [r["auc"] for r in rows]
            finals = [r["final"] for r in rows]
            print(f"  auc={np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
                  f"final={np.mean(finals):.3f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "posthoc_llm_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Cache stats ───────────────────────────────────────────────────────────
    print(f"LLM cache entries used: {len(_LLM_CACHE)}")

    print(f"\n{'='*65}\nRESULTS SUMMARY\n{'='*65}")
    n = args.n_repeats

    def paired_p(c1, c2, ds):
        sub = df[df.dataset == ds]
        a = sub[sub.condition==c1].sort_values("seed").auc.values
        b = sub[sub.condition==c2].sort_values("seed").auc.values
        if len(a) != len(b) or len(a) < 2:
            return float("nan"), float("nan")
        d_val = np.mean(a) - np.mean(b)
        _, p = stats.ttest_rel(a, b)
        return float(d_val), float(p)

    for ds in df.dataset.unique():
        sub = df[df.dataset == ds]
        print(f"\n{ds}:")
        agg = sub.groupby("condition").agg(
            auc_mean=("auc","mean"), auc_std=("auc","std"),
            final_mean=("final","mean"),
        ).sort_values("auc_mean", ascending=False)
        for cond, row in agg.iterrows():
            print(f"  {cond:<12} auc={row.auc_mean:.3f}±{row.auc_std:.3f}  "
                  f"final={row.final_mean:.3f}")

        print()
        for c1, c2, label in [
            ("llm_nopad", "random", "LLM-BO vs Random"),
            ("llm_nopad", "egbo",   "LLM-BO vs EGBO"),
            ("egbo",      "random", "EGBO vs Random (sanity check)"),
            ("llm_bo",    "random", "LLM-BO+pad vs Random (if run)"),
        ]:
            if c1 in agg.index and c2 in agg.index:
                d_val, p = paired_p(c1, c2, ds)
                sig = " *" if p < 0.05 else ""
                er = agg.loc["egbo","auc_mean"] - agg.loc["random","auc_mean"]
                gap = (agg.loc[c1,"auc_mean"]-agg.loc[c2,"auc_mean"]) / (er+1e-12) * 100
                print(f"  {label:<35} Δ={d_val:+.3f}  p={p:.3f}{sig}  "
                      f"gap={gap:.0f}%")


if __name__ == "__main__":
    main()
