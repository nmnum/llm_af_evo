"""
llm_warmstart.py — LLM warm-start module for multi-objective formulation optimization.

The LLM is called ONCE per campaign seed, before any experiments, to generate
an initial set of diverse, mechanistically-reasonable formulations. This
implements the "prior-knowledge routing" architecture recommended by the
mechanistic diagnosis: the LLM contributes domain knowledge up front, then
EGBO runs autonomously.

Three-layer prompt design (for generalisability):
  Layer 1 — Reasoning protocol (domain-agnostic, identical across domains)
  Layer 2 — Domain physics (transferable within a domain family)
  Layer 3 — Specific materials (project-specific, swappable)

Usage:
    from llm_warmstart import llm_warmstart_init, build_warmstart_prompt

    # Real LLM call via Ollama
    X_init, Y_init, forms, meta = llm_warmstart_init(
        disc_oracle, protein="mAb_aggregation", prior_level="L1",
        n_propose=25, n_select=10, model="qwen3:32b",
    )
    # meta["llm_fallback_used"] is True if all 3 real-LLM attempts failed
    # and this run silently fell back to mock formulations.

    # Mock mode (no Ollama needed, deterministic diverse sampling)
    X_init, Y_init, forms, meta = llm_warmstart_init(
        disc_oracle, protein="mAb_aggregation", prior_level="L1",
        n_propose=25, n_select=10, mock=True, seed=42,
    )
"""

import json
import re
import warnings
import numpy as np
from typing import List, Tuple, Dict, Optional

from excipient_oracle import (
    AMINO_ACIDS, SUGARS, SURFACTANTS,
    AA_LIST, SUGAR_LIST, SF_LIST,
    formulation_to_vector, vector_to_formulation,
)


# ── Prior knowledge templates ──────────────────────────────────────────────────
# Layer 1 is domain-agnostic. Layers 2+3 are domain-specific.
# For cross-domain (coatings), only layers 2+3 change.

LAYER1_REASONING = """\
You are designing initial experiments for a multi-objective optimization
campaign. You will track THREE objectives simultaneously:
  Tm (melting temperature, HIGHER = better, conformational stability)
  kD (diffusion interaction parameter, HIGHER = better, colloidal stability)
  viscosity (LOWER = better, manufacturability)

There is no single "best" formulation — you are looking for good TRADE-OFFS
across these three objectives (a Pareto front), not one optimal point.

Propose exactly {n_propose} diverse formulations that:
  1. Target each objective individually — some maximising Tm, some
     maximising kD, some minimising viscosity
  2. Include trade-off formulations that balance two objectives
  3. Include exploration formulations — test excipients you expect to
     be suboptimal, to confirm or refute that expectation
  4. Span the full concentration range for each excipient, not just
     the "obvious" mid-range — include some low-end and some high-end
  5. Vary the categorical choices (different amino acids, different sugars,
     different surfactants) — do not fix one choice and only vary concentration
"""

LAYER2_AGGREGATION = """\
This monoclonal antibody's dominant degradation pathway is AGGREGATION.

General protein formulation science relevant here (Arakawa & Timasheff
preferential-exclusion framework; standard biologics excipient-selection
practice):
  Amino acid excipients differ in how strongly they suppress protein-protein
    self-association. Guanidinium-group-bearing amino acids (arginine) are
    well documented as potent aggregation suppressants, acting through weak,
    transient binding to hydrophobic/aromatic surface patches rather than
    classical preferential exclusion -- this tends to show up as improved
    colloidal/diffusion behaviour more than as a large shift in thermal
    unfolding onset. Simpler amino acids (proline, glycine) give a milder
    version of this effect and are more commonly used as general osmolytes.
  Polyol/sugar excipients (sucrose, trehalose, sorbitol) work by the
    classical preferential-exclusion mechanism -- they raise the free-energy
    cost of unfolding and so tend to raise thermal transition temperatures,
    largely independent of how they affect protein-protein interactions.
    Polyols with known crystallisation risk (mannitol) are generally kept
    below their crystallisation threshold.
  Nonionic surfactants (polysorbates, poloxamers) primarily protect against
    interfacial and shear-induced aggregation rather than bulk thermal or
    colloidal stability, and are typically used at low concentration with
    only a minor direct effect on viscosity. Surfactant benefit is often
    reported as more pronounced when paired with an amino acid that is
    already suppressing bulk self-association, since the two act on
    different aggregation pathways (interfacial vs. bulk).
  Antioxidants (methionine) and metal chelators (EDTA) target oxidative
    degradation, which is not this protein's dominant pathway, so expect
    limited benefit on any of the three objectives here.
  Total dissolved excipient load (amino acid + sugar combined) is a common
    driver of solution viscosity -- there is a real trade-off between
    adding enough stabiliser and keeping the formulation manufacturable.
"""

LAYER2_OXIDATION = """\
This monoclonal antibody's dominant degradation pathway is OXIDATION
(methionine/tryptophan side-chain oxidation), not aggregation.

General protein formulation science relevant here (standard biologics
formulation practice for oxidation-sensitive molecules):
  Sacrificial antioxidants (free methionine) are the standard excipient
    choice for oxidation-prone biologics -- they compete for reactive
    oxygen species ahead of the protein's own susceptible residues. For
    this protein, methionine is expected to matter more than
    aggregation-focused excipients.
  Metal chelators (EDTA) suppress metal-catalysed (Fenton-type) oxidation
    and are commonly paired with a sacrificial antioxidant for
    oxidation-prone molecules.
  Buffering amino acids (histidine) help hold pH in the range that
    minimises oxidation kinetics for many biologics, independent of any
    aggregation-suppression role.
  Polyol/sugar excipients (sucrose, trehalose) still raise thermal
    transition temperature via preferential exclusion -- that mechanism
    is independent of degradation pathway.
  Aggregation-focused excipients (arginine, and the arginine/surfactant
    pairing relevant for aggregation-prone proteins) are expected to be
    LESS impactful here, since aggregation is not this protein's dominant
    liability -- expect a smaller colloidal-stability benefit from
    arginine than you would see on an aggregation-prone protein.
  Total excipient concentration remains a driver of viscosity regardless
    of degradation pathway.
"""

LAYER2_BLANK = """\
No specific domain knowledge is provided about the excipient mechanisms.
Explore the space systematically, varying all available excipients and
concentrations to build a diverse initial dataset.
"""

LAYER2_WRONG = """\
This monoclonal antibody's dominant degradation pathway is OXIDATION.

[Note: the following mechanistic guidance is deliberately mismatched to
this protein's actual pathway -- it describes an aggregation-focused
strategy, which the true pathway above indicates is not the priority here.]
  Guanidinium-bearing amino acids (arginine) are commonly cited as potent
    aggregation suppressants and are treated here as the most important
    excipient for colloidal stability. Use high concentrations.
  Nonionic surfactants are treated here as most valuable when paired with
    arginine -- always combine the two.
  Antioxidants and metal chelators (methionine, EDTA) are treated here as
    addressing a pathway that is not this protein's priority, so minimal
    concentration budget should go toward them.
  Polyol/sugar excipients still raise thermal transition temperature via
    preferential exclusion.
  Total excipient load remains a driver of viscosity.
"""

LAYER3_EXCIPIENTS = """\
Available excipients (choose exactly one from each category):
  Amino acids: {aa_opts}
  Sugars: {sug_opts}
  Surfactants: {sf_opts}
  EDTA: true or false (0.01% if true)
"""

LAYER3_COATINGS = """\
You are optimising a multi-layer coating process with 4 continuous parameters.
All inputs are normalised to [0, 1].

Input parameters and their physical meaning:
  x0 (concentration): precursor concentration (0=low, 1=high)
  x1 (temperature): deposition temperature (0=25C, 1=200C)
  x2 (flow_rate): carrier gas flow rate (0=slow, 1=fast)
  x3 (pressure): chamber pressure (0=vacuum, 1=atmospheric)

Known interactions:
  - High concentration + high temperature -> cracking (avoid x0>0.7 with x1>0.8)
  - Best observed formulations tend to have moderate temperature (0.3-0.7)
"""


def _format_excipient_opts(catalogue: dict, names: list) -> str:
    """Format excipient options for the prompt."""
    parts = []
    for name in names:
        props = catalogue[name]
        parts.append(f"{name}({props['min']}-{props['max']}, step {props['step']})")
    return ", ".join(parts)


def build_warmstart_prompt(
    n_propose: int,
    prior_level: str = "L1",
    protein: str = "mAb_aggregation",
    domain: str = "formulation",
) -> str:
    """
    Build the three-layer warm-start prompt.

    prior_level: "L1" (correct mechanism), "blank" (no knowledge),
                 "wrong" (crossed mechanism — tests robustness)
    protein: determines which layer 2 text is used
    domain: "formulation" or "coatings" — determines layer 3
    """
    layer1 = LAYER1_REASONING.format(n_propose=n_propose)

    if domain == "coatings":
        layer2 = LAYER3_COATINGS  # coatings has combined layer 2+3
        return f"{layer1}\n\n{layer2}\n\nPropose {n_propose} candidate points as JSON."

    # Formulation domain
    if prior_level == "blank":
        layer2 = LAYER2_BLANK
    elif prior_level == "wrong":
        layer2 = LAYER2_WRONG
    else:  # L1
        if "aggregation" in protein:
            layer2 = LAYER2_AGGREGATION
        elif "oxidation" in protein:
            layer2 = LAYER2_OXIDATION
        else:
            layer2 = LAYER2_BLANK

    aa_opts = _format_excipient_opts(AMINO_ACIDS, AA_LIST)
    sug_opts = _format_excipient_opts(SUGARS, SUGAR_LIST)
    sf_opts = _format_excipient_opts(SURFACTANTS, SF_LIST)
    layer3 = LAYER3_EXCIPIENTS.format(aa_opts=aa_opts, sug_opts=sug_opts, sf_opts=sf_opts)

    return f"""{layer1}

{layer2}

{layer3}

For each formulation, state which objective(s) it primarily targets using
the single-letter codes T (Tm), K (kD), V (viscosity) — e.g. ["T","K"] for
a formulation targeting both Tm and kD. State any trade-off it makes in
AT MOST 10 WORDS TOTAL (e.g. "raises viscosity", "weak Tm gain, better kD")
— never exceed 10 words regardless of how many objectives are involved.

Respond with JSON only:
{{
  "candidates": [
    {{"aa": "<name>", "aa_conc": <mM>, "sugar": "<name>", "sugar_conc": <mM>,
      "surfactant": "<name>", "surfactant_conc": <pct>, "edta": <true/false>,
      "targets": ["T"|"K"|"V", ...],
      "tradeoff": "<max 10 words total>"}}
  ]
}}"""


def build_coatings_prompt(n_propose: int) -> str:
    """Build prompt for coatings domain (4D continuous)."""
    return f"""{LAYER1_REASONING.format(n_propose=n_propose)}

{LAYER3_COATINGS}

Propose exactly {n_propose} new candidate points to evaluate.
Each candidate must have 4 values (x0, x1, x2, x3), all in [0, 1].

Respond with valid JSON only:
{{
  "candidates": [
    {{"x": [x0, x1, x2, x3], "targets": ["Tm"|"kD"|"viscosity", ...],
      "tradeoff": "<short phrase>"}}
  ]
}}"""


# ── Parsing and snapping ───────────────────────────────────────────────────────

def _snap(val, props):
    """Snap a concentration value to the valid step grid."""
    val = float(val)
    val = round(val / props["step"]) * props["step"]
    return float(np.clip(val, props["min"], props["max"]))


def _best_match(s: str, opts: list) -> str:
    """Fuzzy-match an excipient name."""
    s = str(s).lower().replace(" ", "").replace("-", "")
    for o in opts:
        o_clean = o.replace(" ", "").replace("-", "")
        if o_clean in s or s in o_clean:
            return o
    return opts[0]


_TARGET_CODE_TO_NAME = {"T": "Tm", "K": "kD", "V": "viscosity"}


def _decode_targets(raw_targets) -> List[str]:
    """Decode single-letter target codes (T/K/V) back to full objective
    names. Falls back to passing through unrecognised values as-is, in
    case the model ignored the code instruction and used full names."""
    if not isinstance(raw_targets, list):
        return []
    return [_TARGET_CODE_TO_NAME.get(str(t).strip().upper(), str(t)) for t in raw_targets]


def parse_llm_formulations(text: str, n_expected: int) -> List[dict]:
    """
    Parse LLM JSON response into a list of formulation dicts.
    Handles <think> tags, markdown fences, trailing commas.
    """
    # Strip <think>...</think> for reasoning models
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Strip markdown fences
    text = re.sub(r'```(?:json)?', '', text).strip().strip('`')
    # Fix trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON array
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                warnings.warn(f"Failed to parse LLM JSON: {text[:200]}")
                return []
        else:
            return []

    raw = parsed.get("candidates", [])
    formulations = []

    for c in raw:
        try:
            aa = _best_match(c.get("aa", ""), AA_LIST)
            sug = _best_match(c.get("sugar", ""), SUGAR_LIST)
            sf = _best_match(c.get("surfactant", ""), SF_LIST)
            form = {
                "aa": aa,
                "aa_conc": _snap(c.get("aa_conc", AMINO_ACIDS[aa]["optimal_conc"]),
                                 AMINO_ACIDS[aa]),
                "sugar": sug,
                "sugar_conc": _snap(c.get("sugar_conc", SUGARS[sug]["optimal_conc"]),
                                    SUGARS[sug]),
                "surfactant": sf,
                "surfactant_conc": _snap(c.get("surfactant_conc",
                                               SURFACTANTS[sf]["optimal_conc"]),
                                         SURFACTANTS[sf]),
                "edta": bool(c.get("edta", False)),
                "targets": _decode_targets(c.get("targets", [])),
                "tradeoff": " ".join(str(c.get("tradeoff", "")).split()[:10]),
            }
            formulations.append(form)
        except Exception:
            continue

    return formulations[:n_expected]


def parse_llm_coatings(text: str, n_expected: int, d: int = 4) -> np.ndarray:
    """Parse LLM JSON response for coatings domain (continuous [0,1]^d)."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'```(?:json)?', '', text).strip().strip('`')
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return np.array([])
        else:
            return np.array([])

    raw = parsed.get("candidates", [])
    candidates = []
    for c in raw:
        x = c.get("x", [])
        if len(x) == d:
            candidates.append(np.clip(np.array(x, dtype=float), 0, 1))

    if not candidates:
        return np.array([])
    return np.array(candidates[:n_expected])


# ── Diversity selection (Kennard-Stone / max-min distance) ────────────────────

def diversity_select(
    candidates_n: np.ndarray,
    n_select: int,
    existing_n: np.ndarray = None,
) -> np.ndarray:
    """
    Select n_select candidates that maximise minimum distance to each other
    and to existing points (max-min diversity, a.k.a. Kennard-Stone).

    Parameters
    ----------
    candidates_n : (N, d) normalised candidate vectors
    n_select : int
    existing_n : (M, d) already-selected/observed points to avoid

    Returns
    -------
    selected_idx : (n_select,) int array — indices into candidates_n
    """
    N = len(candidates_n)
    n_select = min(n_select, N)

    if N <= n_select:
        return np.arange(N)

    # Start from the point farthest from the centroid (or from existing points)
    if existing_n is not None and len(existing_n) > 0:
        # Start from point farthest from existing observations
        dists_to_existing = np.min(
            np.linalg.norm(candidates_n[:, None, :] - existing_n[None, :, :], axis=2),
            axis=1
        )
        first = int(np.argmax(dists_to_existing))
    else:
        # Start from point farthest from centroid
        centroid = candidates_n.mean(axis=0)
        dists = np.linalg.norm(candidates_n - centroid, axis=1)
        first = int(np.argmax(dists))

    selected = [first]
    mask = np.ones(N, dtype=bool)
    mask[first] = False

    for _ in range(n_select - 1):
        if not mask.any():
            break
        avail = np.where(mask)[0]

        # Min distance from each available to any selected
        dists_to_sel = np.min(
            np.linalg.norm(
                candidates_n[avail][:, None, :] - candidates_n[selected][None, :, :],
                axis=2
            ),
            axis=1
        )

        # Also consider distance to existing observations
        if existing_n is not None and len(existing_n) > 0:
            dists_to_obs = np.min(
                np.linalg.norm(
                    candidates_n[avail][:, None, :] - existing_n[None, :, :],
                    axis=2
                ),
                axis=1
            )
            min_dists = np.maximum(dists_to_sel, dists_to_obs)
        else:
            min_dists = dists_to_sel

        best = int(avail[np.argmax(min_dists)])
        selected.append(best)
        mask[best] = False

    return np.array(selected, dtype=int)


# ── Mock LLM (deterministic, no Ollama needed) ────────────────────────────────

def _mock_warmstart_formulations(
    n_propose: int,
    protein: str,
    prior_level: str,
    seed: int,
) -> List[dict]:
    """
    Deterministic mock that generates diverse formulations.
    For L1 prior: biases toward mechanism-relevant excipients.
    For blank: uniform random across all excipients.
    For wrong: biases toward WRONG excipients (tests recovery).
    """
    rng = np.random.default_rng(seed)

    # Determine which AAs are "preferred" based on prior level
    if prior_level == "L1":
        if "aggregation" in protein:
            preferred_aas = ["arginine", "proline"]
            preferred_sugars = ["sucrose", "trehalose", "sorbitol"]
        elif "oxidation" in protein:
            preferred_aas = ["methionine", "histidine"]
            preferred_sugars = ["sucrose", "trehalose"]
        else:
            preferred_aas = AA_LIST
            preferred_sugars = SUGAR_LIST
    elif prior_level == "wrong":
        # Wrong: recommend aggregation excipients for oxidation protein and vice versa
        if "oxidation" in protein:
            preferred_aas = ["arginine", "proline"]  # wrong for oxidation
            preferred_sugars = ["sucrose", "trehalose"]
        else:
            preferred_aas = ["methionine", "histidine"]  # wrong for aggregation
            preferred_sugars = ["sucrose", "trehalose"]
    else:  # blank
        preferred_aas = AA_LIST
        preferred_sugars = SUGAR_LIST

    formulations = []
    for i in range(n_propose):
        # 60% preferred excipients, 40% random (ensures diversity + some bias)
        if rng.random() < 0.6 and prior_level != "blank":
            aa = str(rng.choice(preferred_aas))
            sugar = str(rng.choice(preferred_sugars))
        else:
            aa = str(rng.choice(AA_LIST))
            sugar = str(rng.choice(SUGAR_LIST))

        sf = str(rng.choice(SF_LIST))
        aa_props = AMINO_ACIDS[aa]
        sug_props = SUGARS[sugar]
        sf_props = SURFACTANTS[sf]

        # Span the concentration range: use quantiles, not just mid-range
        quantile = (i % 5) / 4.0  # 0, 0.25, 0.5, 0.75, 1.0
        aa_conc = aa_props["min"] + quantile * (aa_props["max"] - aa_props["min"])
        sugar_conc = sug_props["min"] + ((i + 2) % 5) / 4.0 * (sug_props["max"] - sug_props["min"])
        sf_conc = sf_props["min"] + ((i + 1) % 3) / 2.0 * (sf_props["max"] - sf_props["min"])

        # Add some noise so it's not perfectly gridded
        aa_conc = _snap(aa_conc + rng.normal(0, aa_props["breadth"] * 0.15), aa_props)
        sugar_conc = _snap(sugar_conc + rng.normal(0, sug_props["breadth"] * 0.15), sug_props)
        sf_conc = _snap(sf_conc + rng.normal(0, sf_props["breadth"] * 0.15), sf_props)

        edta = bool(rng.random() < 0.4)

        formulations.append({
            "aa": aa, "aa_conc": aa_conc,
            "sugar": sugar, "sugar_conc": sugar_conc,
            "surfactant": sf, "surfactant_conc": sf_conc,
            "edta": edta,
        })

    return formulations


def _mock_warmstart_coatings(
    n_propose: int,
    seed: int,
    d: int = 4,
) -> np.ndarray:
    """Deterministic diverse sampling for coatings domain."""
    rng = np.random.default_rng(seed)
    # Latin hypercube-like diverse sampling
    candidates = np.zeros((n_propose, d))
    for j in range(d):
        perm = rng.permutation(n_propose)
        candidates[:, j] = (perm + rng.uniform(0, 1, n_propose)) / n_propose
    return np.clip(candidates, 0, 1)


# ── Main entry point ───────────────────────────────────────────────────────────

def llm_warmstart_init(
    disc_oracle,
    protein: str = "mAb_aggregation",
    prior_level: str = "L1",
    n_propose: int = 25,
    n_select: int = 10,
    model: str = "qwen3:32b",
    mock: bool = False,
    seed: int = 42,
    domain: str = "formulation",
) -> Tuple[np.ndarray, np.ndarray, List[dict], dict]:
    """
    Generate initial formulations via LLM warm-start, diversity-select the
    best n_select, and return them as (X_init, Y_init, formulations, meta).

    For formulation domain: returns 16D vectors + (3,) objective vectors.

    Returns
    -------
    X_init : (n_select, d) array — feature vectors of selected formulations
    Y_init : (n_select, 3) array — objective values
    forms : list of n_select formulation dicts
    meta : dict with "llm_fallback_used" (bool) and "llm_retry_count" (int),
        so a run that silently degraded to mock is visible downstream.
    """
    if domain == "formulation":
        return _warmstart_formulation(
            disc_oracle, protein, prior_level, n_propose, n_select,
            model, mock, seed,
        )
    else:
        raise ValueError(f"Use llm_warmstart_init_coatings for domain={domain}")


def _warmstart_formulation(
    disc_oracle,
    protein: str,
    prior_level: str,
    n_propose: int,
    n_select: int,
    model: str,
    mock: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Formulation-domain warm-start."""
    retry_count = 0
    fallback_used = False

    if mock:
        formulations = _mock_warmstart_formulations(
            n_propose, protein, prior_level, seed)
    else:
        prompt = build_warmstart_prompt(
            n_propose=n_propose, prior_level=prior_level, protein=protein,
            domain="formulation",
        )
        import ollama
        text = ""
        formulations = []
        for attempt in range(3):
            retry_count = attempt
            try:
                resp = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system",
                         "content": "You are a pharmaceutical formulation scientist. "
                                    "Respond with valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.4, "num_predict": 4096, "think": False,
                              "seed": seed + attempt},
                )
                text = resp["message"]["content"]
                formulations = parse_llm_formulations(text, n_propose)
                if formulations:
                    break
            except Exception as e:
                if attempt == 2:
                    warnings.warn(f"LLM warm-start failed after 3 attempts: {e}")
                    fallback_used = True
                    formulations = _mock_warmstart_formulations(
                        n_propose, protein, "blank", seed)
                else:
                    continue

    if not formulations:
        warnings.warn("No formulations generated, falling back to mock")
        fallback_used = True
        formulations = _mock_warmstart_formulations(
            n_propose, protein, "blank", seed)

    # Convert to vectors
    vectors = np.array([formulation_to_vector(f) for f in formulations])

    # Diversity selection in normalised 16D space
    # (vectors are already normalised by formulation_to_vector)
    selected_idx = diversity_select(vectors, n_select)
    selected_forms = [formulations[i] for i in selected_idx]
    selected_vectors = vectors[selected_idx]

    # Query the oracle for objective values
    X_init = np.zeros((len(selected_idx), 16))
    Y_init = np.zeros((len(selected_idx), 3))
    for i, vec in enumerate(selected_vectors):
        y, idx = disc_oracle.query_mo(vec)
        X_init[i] = disc_oracle._X_raw[idx]
        Y_init[i] = y

    meta = {"llm_fallback_used": fallback_used, "llm_retry_count": retry_count}
    return X_init, Y_init, selected_forms, meta


def llm_warmstart_init_coatings(
    oracle,
    n_propose: int = 25,
    n_select: int = 10,
    model: str = "qwen3:32b",
    mock: bool = False,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Coatings-domain warm-start. Returns (X_init, y_init, meta) in the oracle's
    native format (continuous [0,1]^d + scalar score). meta contains
    "llm_fallback_used" (bool) and "llm_retry_count" (int).
    """
    d = oracle.bounds().shape[0]
    bounds = oracle.bounds()

    retry_count = 0
    fallback_used = False

    if mock:
        candidates_n = _mock_warmstart_coatings(n_propose, seed, d)
    else:
        prompt = build_coatings_prompt(n_propose)
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
            except Exception as e:
                if attempt == 2:
                    fallback_used = True
                    candidates_n = _mock_warmstart_coatings(n_propose, seed, d)

    if len(candidates_n) == 0:
        fallback_used = True
        candidates_n = _mock_warmstart_coatings(n_propose, seed, d)

    # Diversity select
    selected_idx = diversity_select(candidates_n, n_select)
    selected_n = candidates_n[selected_idx]

    # Convert to raw space and query oracle
    lo, hi = bounds[:, 0], bounds[:, 1]
    selected_raw = selected_n * (hi - lo) + lo

    # Snap to nearest oracle pool point
    all_X = oracle._X_raw
    scaler = oracle._scaler
    X_all_s = scaler.transform(all_X)

    X_init = np.zeros((len(selected_idx), d))
    y_init = np.zeros(len(selected_idx))
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
        y_init[i] = float(oracle._y_raw[chosen])

    meta = {"llm_fallback_used": fallback_used, "llm_retry_count": retry_count}
    return X_init, y_init, meta
