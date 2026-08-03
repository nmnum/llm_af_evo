"""
excipient_stress_test.py — Rigorous stress test for LLM-BO on excipient oracle.

Answers: does LLM domain knowledge accelerate formulation optimisation,
or was the earlier result driven by the explicit optimum hint in the prior?

Design
------
Four prior levels (progressively less information):
  L1: mechanism only — roles and synergies, NO concentrations, NO optima
  L2: mechanism + general ranges — adds typical concentration ranges
  L3: current prompt minus optimum hint — removes "optimum is near X mM" line
  L4: blank — names and ranges only, no mechanism knowledge

Two oracles: mAb_aggregation (arginine+PS80 optimal) and mAb_oxidation
  (methionine+EDTA optimal).

Crossed design: each prior level × each oracle × correct/wrong prior
  mAb_agg oracle + agg prior   → expected LLM win
  mAb_agg oracle + oxid prior  → LLM should fail (wrong mechanism)
  mAb_oxid oracle + oxid prior → expected LLM win
  mAb_oxid oracle + agg prior  → LLM should fail (wrong mechanism)
  All × L4 blank prior         → establishes baseline without domain knowledge

Efficiency optimisations vs previous run:
  - n_repeats=15 (was 20) — still powered for p<0.05 given effect sizes
  - budget=40 (was 80) — matches real lab budget (30-50 experiments)
  - n_batches = (40-10)//4 = 7 (was 17) — 60% fewer LLM calls per seed
  - Shared LLM cache across seeds with identical observation hashes
  - EGBO/random run in parallel (4 workers), LLM sequential

Expected runtime: ~4 hours total (vs ~12 hours previous run)

Usage:
  # Quick test — mock LLM, ~3 minutes
  python excipient_stress_test.py --mock_llm --n_repeats 3

  # Full stress test
  python excipient_stress_test.py --model qwen2.5:72b-instruct --n_workers 4

  # Just the critical crossed design (fastest)
  python excipient_stress_test.py --model qwen2.5:72b-instruct \\
      --prior_levels L1 L4 --crossed_only
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
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).parent))


# ── Prior knowledge levels ────────────────────────────────────────────────────

PRIORS = {

# ─── mAb_aggregation priors ───────────────────────────────────────────────────

"agg_L1": """
You are optimising a protein formulation for an aggregation-prone monoclonal antibody.
The primary degradation pathway is aggregation (intermolecular hydrophobic interactions).

Excipient mechanisms:
  Arginine: prevents protein aggregation by electrostatic charge-shielding of
    hydrophobic patches. The strongest anti-aggregation amino acid.
  Sucrose / trehalose: nearly equivalent stabilisers via preferential exclusion
    from the protein surface (raises the energy cost of unfolding).
  Polysorbate 80: protects against interface-induced aggregation at air-liquid
    and container-closure interfaces. Synergises with arginine for mAbs —
    arginine + polysorbate80 together suppress aggregation better than either alone.
  Methionine: addresses oxidation, NOT aggregation. Minimal benefit here.
  EDTA: chelates metal ions that catalyse oxidative degradation. Moderate benefit.
  Mannitol above ~40 mM: risks crystallisation during freeze-drying — avoid.

Goal: maximise stability score in [0,1]. Higher is better.
No specific concentrations or optimal values are provided — you must infer these
from the campaign observations.
""",

"agg_L2": """
You are optimising a protein formulation for an aggregation-prone monoclonal antibody.
The primary degradation pathway is aggregation.

Excipient mechanisms and typical ranges from pharmaceutical literature:
  Arginine (1-25 mM): anti-aggregation via charge-shielding. Typically used
    at 5-20 mM for mAbs. Higher is not always better due to viscosity.
  Sucrose or trehalose (1-100 mM): stabilisation via preferential exclusion.
    Typically effective at 50-100 mM for lyophilised formulations.
  Polysorbate 80 (0.1-1%): interface protection. Typically 0.01-0.1% suffices.
    Synergy with arginine well-documented for mAbs.
  Methionine (0.1-3 mM): oxidation protection only. Not the priority here.
  EDTA: useful when metal-catalysed oxidation is a concern.
  Avoid: mannitol above 40 mM (crystallisation risk).

Goal: maximise stability score in [0,1].
""",

"agg_L3": """
You are optimising a protein formulation for an aggregation-prone monoclonal antibody.
The primary degradation pathway is aggregation (intermolecular hydrophobic interactions).

Excipient space:
  Amino acid (choose one): arginine, proline, glycine, methionine, histidine
  Sugar (choose one): sucrose, trehalose, sorbitol, mannitol
  Surfactant (choose one): polysorbate80, polysorbate20, poloxamer188
  EDTA: True or False

Key domain knowledge:
  - Arginine (10-20 mM) is the strongest anti-aggregation amino acid.
  - Sucrose or trehalose (50-80 mM) stabilise via preferential exclusion.
  - Polysorbate80 (0.3-0.5%) provides best interface protection for mAbs.
  - Arginine + polysorbate80 synergy is the gold standard for aggregation.
  - Methionine is for oxidation — minimal benefit here.
  - EDTA (0.01%): moderate benefit via metal chelation.
  - Avoid mannitol above 40 mM (crystallisation risk).

Goal: maximise stability score in [0,1].
""",

"agg_L4": """
You are optimising a protein formulation. Maximise stability score in [0,1].

Available excipients and concentration ranges:
  Amino acid (choose one):
    arginine (1-25 mM), proline (1-25 mM), glycine (1-25 mM),
    methionine (0.1-3 mM), histidine (1-10 mM)
  Sugar (choose one):
    sucrose (1-100 mM), trehalose (1-100 mM),
    sorbitol (1-50 mM), mannitol (1-50 mM)
  Surfactant (choose one):
    polysorbate80 (0.1-1%), polysorbate20 (0.1-1%), poloxamer188 (0.1-1%)
  EDTA: True or False

No additional domain knowledge provided. Explore the space systematically.
""",

# ─── mAb_oxidation priors ─────────────────────────────────────────────────────

"oxid_L1": """
You are optimising a protein formulation for an oxidation-prone monoclonal antibody.
The primary degradation pathway is methionine/tryptophan oxidation.

Excipient mechanisms:
  Methionine: sacrificial antioxidant — oxidises preferentially before protein
    methionine/tryptophan residues. The PRIMARY stabiliser for oxidation-prone proteins.
  EDTA: chelates metal ions (iron, copper) that catalyse oxidative reactions via
    Fenton chemistry. EDTA + methionine is the key synergistic combination.
  Sucrose / trehalose: provide conformational stability but do NOT address oxidation.
  Polysorbate 80: protects against interface aggregation. Moderate benefit.
  Arginine: addresses aggregation, NOT oxidation. Minimal benefit here.
  Histidine: buffer capacity, some antioxidant effect at low concentrations.

Goal: maximise stability score in [0,1]. Higher is better.
No specific optimal concentrations provided — infer from observations.
""",

"oxid_L2": """
You are optimising a protein formulation for an oxidation-prone monoclonal antibody.
The primary degradation pathway is oxidation.

Excipient mechanisms and typical ranges:
  Methionine (0.1-3 mM): sacrificial antioxidant. Effective at 0.5-2 mM.
    Low concentration is sufficient — excess methionine adds little benefit.
  EDTA (0.01%): metal chelator. Synergises strongly with methionine.
    The methionine + EDTA combination is widely used for oxidation-prone biologics.
  Sucrose or trehalose (50-100 mM): conformational stabilisers. Useful but
    do not address the primary oxidation pathway directly.
  Polysorbate 80 (0.01-0.1%): interface protection, secondary benefit.
  Arginine: anti-aggregation, NOT anti-oxidation. Not the priority.

Goal: maximise stability score in [0,1].
""",

"oxid_L3": """
You are optimising a protein formulation for an oxidation-prone monoclonal antibody.
The primary degradation pathway is methionine/tryptophan oxidation.

Excipient space:
  Amino acid (choose one): arginine, proline, glycine, methionine, histidine
  Sugar (choose one): sucrose, trehalose, sorbitol, mannitol
  Surfactant (choose one): polysorbate80, polysorbate20, poloxamer188
  EDTA: True or False

Key domain knowledge:
  - Methionine (0.5-2 mM) is the PRIMARY stabiliser — sacrificial oxidation.
  - EDTA (0.01%): chelates metals that catalyse oxidative reactions.
  - Methionine + EDTA synergy is the key combination.
  - Sucrose or trehalose (50-70 mM) for conformational stability.
  - Polysorbate80 (0.3-0.5%) for interface protection.
  - Arginine has minimal benefit for oxidation-prone proteins.

Goal: maximise stability score in [0,1].
""",

"oxid_L4": "agg_L4",   # same blank prior — no mechanism, same ranges

}


# ── Harder oracle (sharper score function) ────────────────────────────────────

def make_harder_oracle(protein: str, n_samples: int = 500, seed: int = 42):
    """
    Build a harder excipient oracle with:
    - Reduced baseline (0.02 instead of 0.10-0.15)
    - Stronger pathway specificity (wrong excipient → score < 0.3)
    - Same interface as DiscreteExcipientOracle
    """
    from excipient_oracle import ExcipientOracle, DiscreteExcipientOracle

    class HarderExcipientOracle(ExcipientOracle):
        def _baseline(self):
            return 0.02  # reduced from 0.10-0.15

        def _aa_contribution(self, aa, aa_conc):
            from excipient_oracle import AMINO_ACIDS
            props = AMINO_ACIDS[aa]
            opt, bw = props["optimal_conc"], props["breadth"]
            # Double the pathway weight to make wrong-aa choices score near 0
            if "anti_aggregation" in props["roles"]:
                peak = 0.50 * self.protein.aggregation_tendency
            elif "antioxidant" in props["roles"]:
                peak = 0.55 * self.protein.oxidation_tendency
            elif "buffer" in props["roles"]:
                peak = 0.20 * (self.protein.denaturation_tendency * 0.5 +
                               self.protein.aggregation_tendency * 0.5)
            else:
                peak = 0.15 * (self.protein.denaturation_tendency * 0.6 +
                               self.protein.aggregation_tendency * 0.4)
            return float(peak * np.exp(-0.5 * ((aa_conc - opt) / bw) ** 2))

        def _synergy(self, aa, surfactant, aa_conc, surfactant_conc):
            from excipient_oracle import SURFACTANTS, AMINO_ACIDS
            props = SURFACTANTS[surfactant]
            if props["synergy_partner"] != aa:
                return 0.0
            aa_opt = AMINO_ACIDS[aa]["optimal_conc"]
            aa_bw  = AMINO_ACIDS[aa]["breadth"]
            sf_opt = props["optimal_conc"]
            sf_bw  = props["breadth"]
            aa_sc = np.exp(-0.5 * ((aa_conc - aa_opt) / aa_bw) ** 2)
            sf_sc = np.exp(-0.5 * ((surfactant_conc - sf_opt) / sf_bw) ** 2)
            # Stronger synergy in harder oracle
            return float(props["synergy_strength"] * 1.5 *
                         self.protein.aggregation_tendency *
                         aa_sc * sf_sc)

    harder = HarderExcipientOracle(protein=protein, seed=seed, noise_level=0.02)
    return harder.make_discrete_oracle(n_samples=n_samples, seed=seed)


# ── LLM call with caching ─────────────────────────────────────────────────────

_STRESS_CACHE: dict = {}


def _obs_hash(X_obs, y_obs):
    arr = np.column_stack([X_obs, y_obs.reshape(-1, 1)])
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]


def llm_propose_formulations(X_obs, y_obs, prior_text, batch_size,
                              model, mock, rng, cache_key=None):
    """
    Ask LLM to propose named formulations. Returns (list_of_dicts, reasoning).
    Uses shared cache across seeds with identical observation histories.
    """
    global _STRESS_CACHE

    if cache_key and cache_key in _STRESS_CACHE:
        return _STRESS_CACHE[cache_key], "", True  # forms, reasoning, cache_hit

    from excipient_oracle import (vector_to_formulation, formulation_to_vector,
                                   AA_LIST, SUGAR_LIST, SF_LIST,
                                   AMINO_ACIDS, SUGARS, SURFACTANTS)

    def fmt(form):
        return (f"{form['aa']} {form['aa_conc']:.1f}mM | "
                f"{form['sugar']} {form['sugar_conc']:.0f}mM | "
                f"{form['surfactant']} {form['surfactant_conc']:.1f}% | "
                f"EDTA={'yes' if form.get('edta') else 'no'}")

    top5 = np.argsort(y_obs)[::-1][:5]
    obs_text = "\n".join(
        f"  {i+1}. {fmt(vector_to_formulation(X_obs[idx]))}  score={y_obs[idx]:.3f}"
        for i, idx in enumerate(top5)
    )

    aa_opts  = ", ".join(f"{a}({AMINO_ACIDS[a]['min']}-{AMINO_ACIDS[a]['max']}mM)"
                         for a in AA_LIST)
    sug_opts = ", ".join(f"{s}({SUGARS[s]['min']}-{SUGARS[s]['max']}mM)"
                         for s in SUGAR_LIST)
    sf_opts  = ", ".join(f"{sf}({SURFACTANTS[sf]['min']}-{SURFACTANTS[sf]['max']}%)"
                         for sf in SF_LIST)

    prompt = f"""{prior_text.strip()}

Campaign: {len(y_obs)} observations. Best score so far: {y_obs.max():.3f}

Top-{len(top5)} formulations (best first):
{obs_text}

Available excipients:
  Amino acids (choose one): {aa_opts}
  Sugars (choose one): {sug_opts}
  Surfactants (choose one): {sf_opts}
  EDTA: true or false

Propose exactly {batch_size} formulations. Respond with JSON only:
{{
  "reasoning": "<brief analysis of what the observations suggest>",
  "candidates": [
    {{"aa": "<name>", "aa_conc": <mM>, "sugar": "<name>",
      "sugar_conc": <mM>, "surfactant": "<name>",
      "surfactant_conc": <pct>, "edta": <true/false>}},
    ...
  ]
}}"""

    if mock:
        # Deterministic mock: propose best known formulation + perturbations
        import copy
        best_idx = int(np.argmax(y_obs))
        base = vector_to_formulation(X_obs[best_idx])
        forms = []
        for i in range(batch_size):
            c = copy.deepcopy(base)
            aa_p = AMINO_ACIDS[c["aa"]]
            c["aa_conc"] = float(np.clip(
                c["aa_conc"] + rng.normal(0, aa_p["breadth"] * 0.2),
                aa_p["min"], aa_p["max"]))
            forms.append(c)
        if cache_key:
            _STRESS_CACHE[cache_key] = forms
        return forms, "mock: perturbing best observed", False

    import ollama
    text = ""
    for attempt in range(3):
        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a pharmaceutical formulation scientist. "
                                "Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.1, "num_predict": 512, "think": False},
            )
            text = resp["message"]["content"]
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            text = re.sub(r'```(?:json)?', '', text).strip().strip('`')
            text = re.sub(r',(\s*[}\]])', r'\1', text)

            parsed = json.loads(text)
            reasoning = parsed.get("reasoning", "")
            raw = parsed.get("candidates", [])

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

            forms = []
            for c in raw:
                try:
                    aa  = best_match(c.get("aa", ""), AA_LIST)
                    sug = best_match(c.get("sugar", ""), SUGAR_LIST)
                    sf  = best_match(c.get("surfactant", ""), SF_LIST)
                    forms.append({
                        "aa": aa,
                        "aa_conc": snap(c.get("aa_conc",
                                             AMINO_ACIDS[aa]["optimal_conc"]),
                                        AMINO_ACIDS[aa]),
                        "sugar": sug,
                        "sugar_conc": snap(c.get("sugar_conc",
                                                  SUGARS[sug]["optimal_conc"]),
                                           SUGARS[sug]),
                        "surfactant": sf,
                        "surfactant_conc": snap(c.get("surfactant_conc",
                                                       SURFACTANTS[sf]["optimal_conc"]),
                                                SURFACTANTS[sf]),
                        "edta": bool(c.get("edta", False)),
                    })
                except Exception:
                    continue

            if forms:
                if cache_key:
                    _STRESS_CACHE[cache_key] = forms
                return forms[:batch_size], reasoning, False

        except Exception as e:
            if attempt == 2:
                warnings.warn(f"LLM failed: {e} | response: {text[:150]}")

    return [], "", False


# ── GP and EGBO (reuse from posthoc_llm_comparison) ─────────────────────────

def _fit_gp(X_obs, y_obs):
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


def _ucb(gp, sc, X_cands, y_obs, progress):
    beta = max(0.5, 5.0 * (1.0 - progress))
    mu, sigma = gp.predict(sc.transform(X_cands), return_std=True)
    return mu + beta * sigma


def _make_pool(X_obs, y_obs, bounds, rng, n=72):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    Xn = (X_obs - lo) / (hi - lo + 1e-12)
    n_ex = n // 2
    top_k = max(1, min(len(y_obs) // 3, n_ex))
    top_idx = np.argsort(y_obs)[-top_k:]
    ex_cands = []
    while len(ex_cands) < n_ex:
        base = Xn[top_idx[rng.integers(top_k)]]
        ex_cands.append(np.clip(base + rng.normal(0, 0.1, d), 0, 1))
    cands_n = np.vstack([ex_cands, rng.random((n - n_ex, d))])
    return cands_n, cands_n * (hi - lo) + lo


def snap_query(cands_raw, oracle, queried):
    all_X = oracle._X_raw
    scaler = oracle._scaler
    X_all_s = scaler.transform(all_X)
    unq = [i for i in range(len(all_X)) if i not in queried]
    if not unq:
        queried.clear(); unq = list(range(len(all_X)))
    new_x, new_y = [], []
    for cand in cands_raw:
        if not unq:
            break
        cs = scaler.transform(cand.reshape(1, -1))[0]
        chosen = unq[int(np.argmin(np.linalg.norm(
            X_all_s[unq] - cs, axis=1)))]
        queried.add(chosen)
        new_x.append(all_X[chosen])
        new_y.append(float(oracle._y_raw[chosen]))
        unq = [i for i in unq if i != chosen]
    return new_x, new_y


# ── Campaign runners ──────────────────────────────────────────────────────────

def run_llm_campaign(oracle, X_init, y_init, budget, prior_text,
                      model, mock, batch_size, seed, prior_key, oracle_key):
    from excipient_oracle import formulation_to_vector
    bounds = oracle.bounds()
    lo, hi = bounds[:, 0], bounds[:, 1]
    rng = np.random.default_rng(seed)

    X_obs, y_obs = X_init.copy(), y_init.copy()
    queried = set()
    scaler = oracle._scaler
    X_all_s = scaler.transform(oracle._X_raw)
    for row in scaler.transform(X_init):
        queried.add(int(np.argmin(np.linalg.norm(X_all_s - row, axis=1))))

    running_best = [float(y_obs.max())] * len(X_init)
    decisions = []

    for b in range((budget - len(X_init)) // batch_size):
        step = len(X_init) + b * batch_size
        progress = step / budget
        obs_h = _obs_hash(X_obs, y_obs)
        cache_key = f"{oracle_key}_{prior_key}_{obs_h}"

        forms, reasoning, cache_hit = llm_propose_formulations(
            X_obs, y_obs, prior_text, batch_size,
            model, mock, rng, cache_key=cache_key,
        )

        if forms:
            cands_raw = np.array([formulation_to_vector(f) for f in forms])
            cands_raw = cands_raw * (hi - lo) + lo
        else:
            # Fallback: GP-UCB
            _, cands_raw = _make_pool(X_obs, y_obs, bounds, rng)
            gp, sc = _fit_gp(X_obs, y_obs)
            scores = _ucb(gp, sc, cands_raw, y_obs, progress)
            cands_raw = cands_raw[np.argsort(scores)[::-1][:batch_size]]

        new_x, new_y = snap_query(cands_raw[:batch_size], oracle, queried)
        if new_x:
            X_obs = np.vstack([X_obs, new_x])
            y_obs = np.append(y_obs, new_y)
        for _ in new_y:
            running_best.append(float(y_obs.max()))
        decisions.append({
            "step": step, "cache_hit": cache_hit,
            "reasoning": reasoning,
            "proposed_aa": [f.get("aa") for f in forms] if forms else [],
            "queried_y": new_y,
        })

    return np.array(running_best), decisions


def run_egbo_campaign(oracle, X_init, y_init, budget, batch_size, seed):
    bounds = oracle.bounds()
    rng = np.random.default_rng(seed)
    X_obs, y_obs = X_init.copy(), y_init.copy()
    queried = set()
    scaler = oracle._scaler
    X_all_s = scaler.transform(oracle._X_raw)
    for row in scaler.transform(X_init):
        queried.add(int(np.argmin(np.linalg.norm(X_all_s - row, axis=1))))

    running_best = [float(y_obs.max())] * len(X_init)
    for b in range((budget - len(X_init)) // batch_size):
        step = len(X_init) + b * batch_size
        cands_n, cands_raw = _make_pool(X_obs, y_obs, bounds, rng)
        gp, sc = _fit_gp(X_obs, y_obs)
        scores = _ucb(gp, sc, cands_raw, y_obs, step / budget)
        selected, remaining = [], list(range(len(cands_n)))
        for _ in range(batch_size):
            if not remaining:
                break
            rem = np.array(remaining)
            acq = scores[rem]
            nov = (np.array([np.min(np.linalg.norm(
                cands_n[selected] - cands_n[i], axis=1)) for i in rem])
                   if selected else np.ones(len(rem)))
            a = (acq-acq.min())/(acq.max()-acq.min()+1e-12)
            n_ = (nov-nov.min())/(nov.max()-nov.min()+1e-12)
            pick = int(rem[np.argmax(0.7*a+0.3*n_)])
            selected.append(pick); remaining.remove(pick)
        new_x, new_y = snap_query(cands_raw[selected], oracle, queried)
        if new_x:
            X_obs = np.vstack([X_obs, new_x])
            y_obs = np.append(y_obs, new_y)
        for _ in new_y:
            running_best.append(float(y_obs.max()))
    return np.array(running_best)


def run_random_campaign(oracle, X_init, y_init, budget, batch_size, seed):
    bounds = oracle.bounds()
    lo, hi = bounds[:, 0], bounds[:, 1]
    d = bounds.shape[0]
    rng = np.random.default_rng(seed)
    X_obs, y_obs = X_init.copy(), y_init.copy()
    queried = set()
    scaler = oracle._scaler
    X_all_s = scaler.transform(oracle._X_raw)
    for row in scaler.transform(X_init):
        queried.add(int(np.argmin(np.linalg.norm(X_all_s - row, axis=1))))
    running_best = [float(y_obs.max())] * len(X_init)
    for _ in range((budget - len(X_init)) // batch_size):
        cands = rng.uniform(lo, hi, (batch_size, d))
        new_x, new_y = snap_query(cands, oracle, queried)
        if new_x:
            X_obs = np.vstack([X_obs, new_x])
            y_obs = np.append(y_obs, new_y)
        for _ in new_y:
            running_best.append(float(y_obs.max()))
    return np.array(running_best)


# ── Shared initialisations ────────────────────────────────────────────────────

def make_shared_inits(oracle, n_repeats, n_init, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    all_X = oracle._X_raw
    inits = []
    for _ in range(n_repeats):
        idx = rng.choice(len(all_X), n_init, replace=False)
        X_i = all_X[idx]
        y_i = oracle._y_raw[idx]
        inits.append((X_i, y_i))
    return inits


# ── Parallel seed runner ──────────────────────────────────────────────────────

def _run_seed(task):
    (seed_idx, oracle_key, protein, prior_key, prior_text, condition,
     budget, n_init, batch_size, model, mock, out_dir_str) = task

    oracle = make_harder_oracle(protein, n_samples=500, seed=42)
    inits  = make_shared_inits(oracle, seed_idx + 1, n_init, rng_seed=42)
    X_init, y_init = inits[seed_idx]
    gb = oracle.global_best()

    if condition == "random":
        curve = run_random_campaign(oracle, X_init, y_init, budget,
                                     batch_size, seed_idx)
        decisions = []
    elif condition == "egbo":
        curve = run_egbo_campaign(oracle, X_init, y_init, budget,
                                   batch_size, seed_idx)
        decisions = []
    else:  # llm
        curve, decisions = run_llm_campaign(
            oracle, X_init, y_init, budget, prior_text,
            model, mock, batch_size, seed_idx, prior_key, oracle_key)

    auc   = float((curve / gb).mean())
    final = float(curve[-1] / gb)

    log = {"oracle": oracle_key, "prior": prior_key, "condition": condition,
           "seed": seed_idx, "auc": auc, "final": final,
           "decisions": decisions}

    out_dir = pathlib.Path(out_dir_str)
    cond_dir = out_dir / oracle_key / prior_key / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    with open(cond_dir / f"seed_{seed_idx:03d}.json", "w") as f:
        json.dump(log, f, indent=2)

    return {"oracle": oracle_key, "prior": prior_key, "condition": condition,
            "seed": seed_idx, "auc": auc, "final": final}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",      default="results_stress_test")
    parser.add_argument("--n_repeats",    type=int, default=15)
    parser.add_argument("--budget",       type=int, default=40)
    parser.add_argument("--n_init",       type=int, default=10)
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--model",        default="qwen2.5:72b-instruct")
    parser.add_argument("--mock_llm",     action="store_true")
    parser.add_argument("--n_workers",    type=int, default=4)
    parser.add_argument("--prior_levels", nargs="+",
                        default=["L1","L2","L3","L4"],
                        choices=["L1","L2","L3","L4"])
    parser.add_argument("--crossed_only", action="store_true",
                        help="Only run crossed design (wrong prior) + L1 + L4")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build experiment grid
    oracles = [
        ("mAb_aggregation", "mAb_aggregation"),
        ("mAb_oxidation",   "mAb_oxidation"),
    ]

    # Prior map: oracle_key × level → prior_key
    prior_map = {
        ("mAb_aggregation", "L1"): "agg_L1",
        ("mAb_aggregation", "L2"): "agg_L2",
        ("mAb_aggregation", "L3"): "agg_L3",
        ("mAb_aggregation", "L4"): "agg_L4",
        ("mAb_oxidation",   "L1"): "oxid_L1",
        ("mAb_oxidation",   "L2"): "oxid_L2",
        ("mAb_oxidation",   "L3"): "oxid_L3",
        ("mAb_oxidation",   "L4"): "agg_L4",  # same blank prior
        # Crossed: wrong protein prior
        ("mAb_aggregation", "WRONG"): "oxid_L1",
        ("mAb_oxidation",   "WRONG"): "agg_L1",
    }

    levels = args.prior_levels
    if args.crossed_only:
        levels = ["L1", "L4", "WRONG"]

    all_rows = []

    for oracle_key, protein in oracles:
        oracle = make_harder_oracle(protein, n_samples=500, seed=42)
        gb = oracle.global_best()
        print(f"\n{'='*60}")
        print(f"Oracle: {oracle_key}  N=500  budget={args.budget}  "
              f"global_best={gb:.4f}")

        # First run egbo and random (fast, parallel) for all levels at once
        # (they don't depend on prior level)
        print(f"  Running egbo + random ({args.n_repeats} seeds, "
              f"{args.n_workers} workers)...")
        baseline_tasks = [
            (seed_idx, oracle_key, protein, "baseline", "", cond,
             args.budget, args.n_init, args.batch_size,
             args.model, args.mock_llm, str(out_dir))
            for cond in ["egbo", "random"]
            for seed_idx in range(args.n_repeats)
        ]
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futures = {ex.submit(_run_seed, t): t for t in baseline_tasks}
            for fut in as_completed(futures):
                row = fut.result()
                all_rows.append(row)
                print(".", end="", flush=True)
        print()

        # Then run LLM for each prior level (sequential — Ollama concurrency)
        for level in levels:
            prior_key = prior_map.get((oracle_key, level))
            if prior_key is None:
                continue
            prior_text = PRIORS.get(prior_key, "")
            if prior_text == "agg_L4":
                prior_text = PRIORS["agg_L4"]

            label = f"llm_{level}"
            label_str = f"WRONG prior ({prior_key})" if level == "WRONG" else f"L{level[-1] if len(level)>1 else level}"
            print(f"  {label} ({label_str})...", end="", flush=True)

            for seed_idx in range(args.n_repeats):
                task = (seed_idx, oracle_key, protein, prior_key, prior_text,
                        "llm", args.budget, args.n_init, args.batch_size,
                        args.model, args.mock_llm, str(out_dir))
                row = _run_seed(task)
                row["condition"] = label
                all_rows.append(row)
                print(".", end="", flush=True)
            print()

    # ── Summary ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "stress_test_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCache entries used: {len(_STRESS_CACHE)}")
    print(f"Saved: {csv_path}")

    print(f"\n{'='*70}")
    print("STRESS TEST RESULTS")
    print(f"{'='*70}")
    print("Key: * = p<0.05 paired t-test vs EGBO")
    print()

    n = args.n_repeats

    def pt(a, b):
        if len(a) != len(b) or len(a) < 2:
            return float("nan"), float("nan")
        d = np.mean(a) - np.mean(b)
        _, p = stats.ttest_rel(np.array(a), np.array(b))
        return float(d), float(p)

    for oracle_key, _ in oracles:
        sub = df[df.oracle == oracle_key]
        if sub.empty:
            continue
        egbo_aucs   = sub[sub.condition=="egbo"].sort_values("seed").auc.values
        random_aucs = sub[sub.condition=="random"].sort_values("seed").auc.values
        print(f"\nOracle: {oracle_key}")
        print(f"  {'Condition':<20} {'AUC':>7} {'±SD':>6}  {'vs EGBO':>10}  {'vs Random':>10}")
        print(f"  {'-'*58}")

        for cond_label in ["egbo","random"] + [f"llm_{l}" for l in levels]:
            sub_c = sub[sub.condition==cond_label].sort_values("seed")
            if sub_c.empty:
                continue
            aucs = sub_c.auc.values
            m, s = float(aucs.mean()), float(aucs.std())
            d_e, p_e = pt(list(aucs), list(egbo_aucs))
            d_r, p_r = pt(list(aucs), list(random_aucs))
            sig_e = "*" if (not np.isnan(p_e) and p_e < 0.05) else " "
            sig_r = "*" if (not np.isnan(p_r) and p_r < 0.05) else " "
            print(f"  {cond_label:<20} {m:>7.3f} {s:>6.3f}  "
                  f"{d_e:>+8.3f}{sig_e}  {d_r:>+8.3f}{sig_r}")

    print()
    print("INTERPRETATION GUIDE")
    print("  llm_L1 > egbo *:   mechanism-only knowledge helps")
    print("  llm_L4 ~ egbo:     blank prior = no advantage → mechanism is active ingredient")
    print("  llm_WRONG < egbo:  wrong mechanism hurts → LLM reasons about pathway")
    print("  llm_L1 ~ llm_L3:   removing optimum hint doesn't hurt → not reading the answer")
    print("  llm_L1 >> llm_L4:  knowledge gradient confirms domain knowledge is the driver")


if __name__ == "__main__":
    main()
