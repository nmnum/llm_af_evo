"""
excipient_oracle.py — Synthetic pharmaceutical formulation oracle.

Models protein stability as a function of excipient composition with:
  - Realistic dose-response curves per excipient class
  - Synergistic interactions (arginine+polysorbate80, sucrose+methionine)
  - Antagonistic interactions (mannitol crystallisation at high concentration)
  - Multiple degradation pathways (aggregation, oxidation, denaturation)
  - Flat regions (wrong amino acid for the protein's degradation mode)
  - Sharp optima (synergy combinations)
  - Viscosity penalty for high total solids
  - EDTA conditional benefit (chelation pathway matters)
  - Reproducible with a fixed random seed for inter-experiment comparisons

The oracle supports two modes:
  1. Discrete lookup (NNOracle-compatible): sample N formulations via LHS,
     store as a finite pool, return nearest-neighbour measurements.
     Use this for fair comparison with EGBO/random (same infrastructure).
  2. Continuous evaluation: call score() directly for any valid formulation.
     Use this for ground-truth landscape analysis.

Input space (structured, not raw coordinates):
  Formulation = {
    aa:             str   — one of {arginine, proline, glycine, methionine, histidine}
    aa_conc:        float — mM, within excipient range
    sugar:          str   — one of {sucrose, trehalose, sorbitol, mannitol}
    sugar_conc:     float — mM, within excipient range
    surfactant:     str   — one of {polysorbate80, polysorbate20, poloxamer188}
    surfactant_conc:float — % w/v, within excipient range
    edta:           bool  — include 0.01% EDTA or not
  }

Output: stability_score in [0, 1], where 1.0 = perfect stability.

Usage:
    from excipient_oracle import ExcipientOracle, formulation_to_vector, vector_to_formulation

    oracle = ExcipientOracle(seed=42)
    # Direct evaluation
    form = {"aa": "arginine", "aa_conc": 15.0, "sugar": "sucrose",
            "sugar_conc": 70.0, "surfactant": "polysorbate80",
            "surfactant_conc": 0.4, "edta": True}
    score = oracle.score(form)

    # NNOracle-compatible discrete pool
    pool_oracle = oracle.make_discrete_oracle(n_samples=500, seed=42)
    # pool_oracle.query(x) → float, pool_oracle.global_best() → float
    # pool_oracle.bounds() → np.ndarray, pool_oracle._X_raw, pool_oracle._y_raw
"""

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


# ── Excipient catalogue ────────────────────────────────────────────────────────

AMINO_ACIDS = {
    "arginine": {
        "min": 1.0, "max": 25.0, "step": 1.0,
        "roles": ["anti_aggregation", "solubiliser", "viscosity_reducer"],
        "optimal_conc": 15.0,   # mM — peak anti-aggregation effect
        "breadth": 8.0,         # Gaussian width around optimal
    },
    "proline": {
        "min": 1.0, "max": 25.0, "step": 1.0,
        "roles": ["osmolyte", "anti_aggregation", "cryoprotectant"],
        "optimal_conc": 12.0,
        "breadth": 6.0,
    },
    "glycine": {
        "min": 1.0, "max": 25.0, "step": 1.0,
        "roles": ["stabiliser", "tonicity_agent", "cryoprotectant"],
        "optimal_conc": 18.0,
        "breadth": 7.0,
    },
    "methionine": {
        "min": 0.1, "max": 3.0, "step": 0.1,
        "roles": ["antioxidant", "oxidation_protection"],
        "optimal_conc": 1.0,    # low concentration optimal
        "breadth": 0.8,
    },
    "histidine": {
        "min": 1.0, "max": 10.0, "step": 1.0,
        "roles": ["buffer", "stabiliser"],
        "optimal_conc": 6.0,
        "breadth": 3.0,
    },
}

SUGARS = {
    "sucrose": {
        "min": 1.0, "max": 100.0, "step": 1.0,
        "roles": ["stabiliser", "cryoprotectant", "preferential_exclusion"],
        "optimal_conc": 70.0,
        "breadth": 20.0,
        "crystallisation_risk": False,
    },
    "trehalose": {
        "min": 1.0, "max": 100.0, "step": 1.0,
        "roles": ["stabiliser", "cryoprotectant", "preferential_exclusion"],
        "optimal_conc": 65.0,
        "breadth": 22.0,
        "crystallisation_risk": False,
    },
    "sorbitol": {
        "min": 1.0, "max": 50.0, "step": 1.0,
        "roles": ["stabiliser", "tonicity_agent", "osmolyte"],
        "optimal_conc": 35.0,
        "breadth": 12.0,
        "crystallisation_risk": False,
    },
    "mannitol": {
        "min": 1.0, "max": 50.0, "step": 1.0,
        "roles": ["tonicity_agent", "lyoprotectant"],
        "optimal_conc": 25.0,   # peaks early, crystallisation risk above 40
        "breadth": 10.0,
        "crystallisation_risk": True,
        "crystallisation_threshold": 40.0,
    },
}

SURFACTANTS = {
    "polysorbate80": {
        "min": 0.1, "max": 1.0, "step": 0.1,
        "roles": ["anti_adsorption", "anti_interface_aggregation", "shear_protection"],
        "optimal_conc": 0.4,
        "breadth": 0.25,
        "synergy_partner": "arginine",
        "synergy_strength": 0.18,
    },
    "polysorbate20": {
        "min": 0.1, "max": 1.0, "step": 0.1,
        "roles": ["anti_adsorption", "anti_interface_aggregation"],
        "optimal_conc": 0.35,
        "breadth": 0.25,
        "synergy_partner": "arginine",
        "synergy_strength": 0.12,  # weaker than PS80
    },
    "poloxamer188": {
        "min": 0.1, "max": 1.0, "step": 0.1,
        "roles": ["anti_adsorption", "shear_protection"],
        "optimal_conc": 0.5,
        "breadth": 0.3,
        "synergy_partner": None,
        "synergy_strength": 0.0,
    },
}


# ── Oracle core ────────────────────────────────────────────────────────────────

@dataclass
class ProteinProfile:
    """
    Protein-specific degradation mode that determines which excipients matter.
    Different proteins need different formulation strategies.
    """
    name: str
    aggregation_tendency: float   # 0-1, how prone to aggregation
    oxidation_tendency: float     # 0-1, how prone to oxidation (Met/Trp)
    denaturation_tendency: float  # 0-1, how thermally labile
    metal_sensitivity: float      # 0-1, how much metal-catalysed oxidation matters
    # Relative weights determine which pathway dominates
    noise_level: float = 0.03    # measurement noise (sigma)


# Canonical protein profiles — these are the "true" unknowns in the oracle
PROTEIN_PROFILES = {
    "mAb_aggregation": ProteinProfile(
        name="mAb_aggregation",
        aggregation_tendency=0.85,
        oxidation_tendency=0.20,
        denaturation_tendency=0.30,
        metal_sensitivity=0.15,
    ),
    "mAb_oxidation": ProteinProfile(
        name="mAb_oxidation",
        aggregation_tendency=0.25,
        oxidation_tendency=0.80,
        denaturation_tendency=0.25,
        metal_sensitivity=0.60,
    ),
    "enzyme_labile": ProteinProfile(
        name="enzyme_labile",
        aggregation_tendency=0.40,
        oxidation_tendency=0.30,
        denaturation_tendency=0.75,
        metal_sensitivity=0.20,
    ),
    "mixed": ProteinProfile(
        name="mixed",
        aggregation_tendency=0.55,
        oxidation_tendency=0.45,
        denaturation_tendency=0.45,
        metal_sensitivity=0.35,
    ),
}


class ExcipientOracle:
    """
    Synthetic pharmaceutical formulation oracle.

    The protein profile is fixed at construction and unknown to the optimiser —
    this is the key challenge: the LLM must infer which degradation pathway
    dominates from early observations and adjust its formulation strategy.
    """

    def __init__(self, protein: str = "mAb_aggregation", seed: int = 42,
                 noise_level: float = 0.03):
        self.protein = PROTEIN_PROFILES[protein]
        self.protein.noise_level = noise_level
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self._protein_name = protein

    def _aa_contribution(self, aa: str, aa_conc: float) -> float:
        """Gaussian dose-response for amino acid, weighted by protein profile."""
        props = AMINO_ACIDS[aa]
        opt = props["optimal_conc"]
        bw  = props["breadth"]
        # Peak response depends on relevance to degradation mode
        if "anti_aggregation" in props["roles"]:
            peak = 0.35 * self.protein.aggregation_tendency
        elif "antioxidant" in props["roles"]:
            peak = 0.40 * self.protein.oxidation_tendency
        elif "buffer" in props["roles"]:
            peak = 0.20 * (self.protein.denaturation_tendency * 0.5 +
                           self.protein.aggregation_tendency * 0.5)
        else:
            peak = 0.18 * (self.protein.denaturation_tendency * 0.6 +
                           self.protein.aggregation_tendency * 0.4)
        return float(peak * np.exp(-0.5 * ((aa_conc - opt) / bw) ** 2))

    def _sugar_contribution(self, sugar: str, sugar_conc: float) -> float:
        """Saturating dose-response for sugars, with crystallisation penalty."""
        props = SUGARS[sugar]
        opt = props["optimal_conc"]
        bw  = props["breadth"]
        base_peak = 0.30 * (self.protein.denaturation_tendency * 0.5 +
                             self.protein.aggregation_tendency * 0.5)
        score = float(base_peak * np.exp(-0.5 * ((sugar_conc - opt) / bw) ** 2))

        # Mannitol crystallisation penalty above threshold
        if props["crystallisation_risk"]:
            threshold = props["crystallisation_threshold"]
            if sugar_conc > threshold:
                penalty = 0.15 * (sugar_conc - threshold) / (props["max"] - threshold)
                score -= penalty
        return max(0.0, score)

    def _surfactant_contribution(self, surfactant: str,
                                  surfactant_conc: float) -> float:
        """Gaussian dose-response for surfactant."""
        props = SURFACTANTS[surfactant]
        opt = props["optimal_conc"]
        bw  = props["breadth"]
        peak = 0.20 * self.protein.aggregation_tendency
        return float(peak * np.exp(-0.5 * ((surfactant_conc - opt) / bw) ** 2))

    def _synergy(self, aa: str, surfactant: str,
                  aa_conc: float, surfactant_conc: float) -> float:
        """
        Arginine + polysorbate80/20 synergy: both must be near-optimal.
        Classic mAb stabilisation strategy — worth modelling explicitly.
        """
        props = SURFACTANTS[surfactant]
        if props["synergy_partner"] != aa:
            return 0.0
        # Both must be in good range for synergy to fire
        aa_opt = AMINO_ACIDS[aa]["optimal_conc"]
        aa_bw  = AMINO_ACIDS[aa]["breadth"]
        sf_opt = props["optimal_conc"]
        sf_bw  = props["breadth"]
        aa_score = np.exp(-0.5 * ((aa_conc - aa_opt) / aa_bw) ** 2)
        sf_score = np.exp(-0.5 * ((surfactant_conc - sf_opt) / sf_bw) ** 2)
        return float(props["synergy_strength"] *
                     self.protein.aggregation_tendency *
                     aa_score * sf_score)

    def _edta_contribution(self, edta: bool) -> float:
        """EDTA is conditionally useful — only when metal-catalysed oxidation matters."""
        if not edta:
            return 0.0
        return float(0.08 * self.protein.metal_sensitivity)

    def _viscosity_penalty(self, aa_conc: float, sugar_conc: float) -> float:
        """
        High total solids → viscosity → manufacturability penalty.
        Relevant for high-arginine + high-sucrose combinations.
        """
        # Approximate total osmolarity contribution
        total = aa_conc / 25.0 + sugar_conc / 100.0  # normalised
        if total > 1.2:
            return float(0.10 * (total - 1.2))
        return 0.0

    def _baseline(self) -> float:
        """Protein stability without any excipients — context for normalisation."""
        # Aggregation-prone proteins are less stable at baseline
        return float(0.10 + 0.05 * (1 - self.protein.aggregation_tendency))

    def score(self, formulation: dict,
              add_noise: bool = True) -> float:
        """
        Evaluate a formulation. Returns stability score in [0, 1].

        Parameters
        ----------
        formulation : dict with keys aa, aa_conc, sugar, sugar_conc,
                      surfactant, surfactant_conc, edta
        add_noise   : add Gaussian measurement noise (sigma = protein.noise_level)
        """
        aa          = formulation["aa"]
        aa_conc     = float(formulation["aa_conc"])
        sugar       = formulation["sugar"]
        sugar_conc  = float(formulation["sugar_conc"])
        surfactant  = formulation["surfactant"]
        sf_conc     = float(formulation["surfactant_conc"])
        edta        = bool(formulation.get("edta", False))

        # Validate inputs
        for name, cat, props in [
            (aa, "amino acid", AMINO_ACIDS),
            (sugar, "sugar", SUGARS),
            (surfactant, "surfactant", SURFACTANTS),
        ]:
            if name not in props:
                raise ValueError(f"Unknown {cat}: {name}")

        score = (
            self._baseline()
            + self._aa_contribution(aa, aa_conc)
            + self._sugar_contribution(sugar, sugar_conc)
            + self._surfactant_contribution(surfactant, sf_conc)
            + self._synergy(aa, surfactant, aa_conc, sf_conc)
            + self._edta_contribution(edta)
            - self._viscosity_penalty(aa_conc, sugar_conc)
        )
        score = float(np.clip(score, 0.0, 1.0))

        if add_noise:
            noise = self.rng.normal(0, self.protein.noise_level)
            score = float(np.clip(score + noise, 0.0, 1.0))

        return score

    def global_best_formulation(self) -> tuple[dict, float]:
        """Return the theoretical best formulation for this protein (no noise)."""
        # For each protein, the best is typically the strongest synergy combination
        bests = []
        for aa in AMINO_ACIDS:
            aa_opt = AMINO_ACIDS[aa]["optimal_conc"]
            for sugar in SUGARS:
                s_opt = SUGARS[sugar]["optimal_conc"]
                for sf in SURFACTANTS:
                    sf_opt = SURFACTANTS[sf]["optimal_conc"]
                    for edta in [False, True]:
                        form = {"aa": aa, "aa_conc": aa_opt, "sugar": sugar,
                                "sugar_conc": s_opt, "surfactant": sf,
                                "surfactant_conc": sf_opt, "edta": edta}
                        s = self.score(form, add_noise=False)
                        bests.append((s, form))
        bests.sort(key=lambda x: -x[0])
        return bests[0][1], bests[0][0]

    def make_discrete_oracle(self, n_samples: int = 500,
                              seed: int = 42) -> "DiscreteExcipientOracle":
        """
        Sample n_samples formulations via stratified random sampling and
        return an NNOracle-compatible discrete pool.
        """
        return DiscreteExcipientOracle.build(self, n_samples=n_samples, seed=seed)


# ── Vector encoding ────────────────────────────────────────────────────────────

AA_LIST   = list(AMINO_ACIDS.keys())     # 5
SUGAR_LIST = list(SUGARS.keys())         # 4
SF_LIST   = list(SURFACTANTS.keys())     # 3

# Feature vector layout:
#  [0-4]  aa one-hot (5)
#  [5]    aa_conc (1)
#  [6-9]  sugar one-hot (4)
#  [10]   sugar_conc (1)
#  [11-13] surfactant one-hot (3)
#  [14]   surfactant_conc (1)
#  [15]   edta (1)
# Total: 16 dimensions
FEATURE_DIM = 16


def formulation_to_vector(form: dict) -> np.ndarray:
    """Convert structured formulation dict to 16D feature vector."""
    v = np.zeros(FEATURE_DIM, dtype=float)

    # AA one-hot
    aa_idx = AA_LIST.index(form["aa"])
    v[aa_idx] = 1.0
    # AA concentration (normalised to [0,1] within its range)
    aa_props = AMINO_ACIDS[form["aa"]]
    v[5] = (float(form["aa_conc"]) - aa_props["min"]) / (
        aa_props["max"] - aa_props["min"])

    # Sugar one-hot
    s_idx = SUGAR_LIST.index(form["sugar"])
    v[6 + s_idx] = 1.0
    # Sugar concentration
    s_props = SUGARS[form["sugar"]]
    v[10] = (float(form["sugar_conc"]) - s_props["min"]) / (
        s_props["max"] - s_props["min"])

    # Surfactant one-hot
    sf_idx = SF_LIST.index(form["surfactant"])
    v[11 + sf_idx] = 1.0
    # Surfactant concentration
    sf_props = SURFACTANTS[form["surfactant"]]
    v[14] = (float(form["surfactant_conc"]) - sf_props["min"]) / (
        sf_props["max"] - sf_props["min"])

    # EDTA
    v[15] = 1.0 if form.get("edta", False) else 0.0

    return v


def vector_to_formulation(v: np.ndarray) -> dict:
    """Convert 16D feature vector back to structured formulation dict."""
    aa_idx = int(np.argmax(v[0:5]))
    aa = AA_LIST[aa_idx]
    aa_props = AMINO_ACIDS[aa]
    aa_conc = float(v[5]) * (aa_props["max"] - aa_props["min"]) + aa_props["min"]
    # Snap to valid step
    aa_conc = round(aa_conc / aa_props["step"]) * aa_props["step"]
    aa_conc = float(np.clip(aa_conc, aa_props["min"], aa_props["max"]))

    s_idx = int(np.argmax(v[6:10]))
    sugar = SUGAR_LIST[s_idx]
    s_props = SUGARS[sugar]
    sugar_conc = float(v[10]) * (s_props["max"] - s_props["min"]) + s_props["min"]
    sugar_conc = round(sugar_conc / s_props["step"]) * s_props["step"]
    sugar_conc = float(np.clip(sugar_conc, s_props["min"], s_props["max"]))

    sf_idx = int(np.argmax(v[11:14]))
    sf = SF_LIST[sf_idx]
    sf_props = SURFACTANTS[sf]
    sf_conc = float(v[14]) * (sf_props["max"] - sf_props["min"]) + sf_props["min"]
    sf_conc = round(sf_conc / sf_props["step"], 1) * sf_props["step"]
    sf_conc = float(np.clip(sf_conc, sf_props["min"], sf_props["max"]))

    edta = bool(v[15] > 0.5)

    return {"aa": aa, "aa_conc": aa_conc, "sugar": sugar, "sugar_conc": sugar_conc,
            "surfactant": sf, "surfactant_conc": sf_conc, "edta": edta}


# ── Discrete oracle (NNOracle-compatible) ──────────────────────────────────────

class DiscreteExcipientOracle:
    """
    NNOracle-compatible interface for the excipient formulation oracle.
    Samples a finite pool of formulations, supports nearest-neighbour lookup.
    Compatible with posthoc_llm_comparison.py infrastructure.
    """

    def __init__(self, X_raw: np.ndarray, y_raw: np.ndarray,
                 forms: list, scaler: StandardScaler):
        self._X_raw = X_raw       # (N, 16) feature vectors
        self._y_raw = y_raw       # (N,) stability scores
        self._forms = forms       # list of N formulation dicts
        self._scaler = scaler
        self._queried = set()

    @classmethod
    def build(cls, oracle: ExcipientOracle, n_samples: int = 500,
               seed: int = 42) -> "DiscreteExcipientOracle":
        """
        Sample n_samples formulations by stratified random sampling:
        - Enumerate all (aa × sugar × surfactant × edta) = 5×4×3×2 = 120 combos
        - For each combo, sample multiple concentration values
        """
        rng = np.random.default_rng(seed)
        forms = []

        # Stratified: ensure all excipient combos are represented
        for aa, sugar, sf, edta in itertools.product(
            AA_LIST, SUGAR_LIST, SF_LIST, [False, True]
        ):
            n_conc_samples = max(1, n_samples // 120)
            for _ in range(n_conc_samples):
                aa_props = AMINO_ACIDS[aa]
                s_props  = SUGARS[sugar]
                sf_props = SURFACTANTS[sf]

                aa_conc = float(rng.uniform(aa_props["min"], aa_props["max"]))
                aa_conc = round(aa_conc / aa_props["step"]) * aa_props["step"]

                s_conc  = float(rng.uniform(s_props["min"], s_props["max"]))
                s_conc  = round(s_conc / s_props["step"]) * s_props["step"]

                sf_conc = float(rng.uniform(sf_props["min"], sf_props["max"]))
                sf_conc = round(sf_conc / sf_props["step"], 1) * sf_props["step"]

                forms.append({"aa": aa, "aa_conc": aa_conc, "sugar": sugar,
                              "sugar_conc": s_conc, "surfactant": sf,
                              "surfactant_conc": sf_conc, "edta": edta})

        # Fill remainder with fully random samples
        while len(forms) < n_samples:
            aa  = rng.choice(AA_LIST)
            sug = rng.choice(SUGAR_LIST)
            sf  = rng.choice(SF_LIST)
            edta = bool(rng.random() > 0.5)
            aa_p = AMINO_ACIDS[aa]; s_p = SUGARS[sug]; sf_p = SURFACTANTS[sf]
            forms.append({
                "aa": aa,
                "aa_conc": round(float(rng.uniform(aa_p["min"], aa_p["max"]))
                                 / aa_p["step"]) * aa_p["step"],
                "sugar": sug,
                "sugar_conc": round(float(rng.uniform(s_p["min"], s_p["max"]))
                                    / s_p["step"]) * s_p["step"],
                "surfactant": sf,
                "surfactant_conc": round(float(rng.uniform(sf_p["min"],
                                               sf_p["max"])) / sf_p["step"],
                                         1) * sf_p["step"],
                "edta": edta,
            })

        # Score all formulations
        X_raw = np.array([formulation_to_vector(f) for f in forms])
        y_raw = np.array([oracle.score(f, add_noise=True) for f in forms])

        # Fit scaler for nearest-neighbour lookup
        scaler = StandardScaler()
        scaler.fit(X_raw)

        return cls(X_raw, y_raw, forms, scaler)

    def bounds(self) -> np.ndarray:
        """Returns (16, 2) bounds array — all features in [0,1] after normalisation."""
        return np.column_stack([np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM)])

    def global_best(self) -> float:
        return float(self._y_raw.max())

    def global_worst(self) -> float:
        return float(self._y_raw.min())

    def query(self, x: np.ndarray) -> float:
        """Query by nearest-neighbour in scaled feature space.

        Returns a plain float (NNOracle-compatible interface).
        The internal _queried set tracks which pool entries have been used
        so repeated queries return the next-nearest unqueried neighbour.
        """
        x_s = self._scaler.transform(x.reshape(1, -1))[0]
        X_s = self._scaler.transform(self._X_raw)
        unqueried = [i for i in range(len(self._X_raw)) if i not in self._queried]
        if not unqueried:
            unqueried = list(range(len(self._X_raw)))
        dists = np.linalg.norm(X_s[unqueried] - x_s, axis=1)
        chosen = unqueried[int(np.argmin(dists))]
        self._queried.add(chosen)
        return float(self._y_raw[chosen])  # plain float, not a tuple

    def get_formulation(self, idx: int) -> dict:
        """Return the formulation dict for a given index."""
        return self._forms[idx]

    def __len__(self):
        return len(self._X_raw)


# ── Landscape analysis ─────────────────────────────────────────────────────────

def analyse_landscape(protein: str = "mAb_aggregation",
                       n_samples: int = 2000) -> dict:
    """
    Characterise the landscape for a given protein profile.
    Returns statistics useful for the GP trust diagnostic.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    oracle_cont = ExcipientOracle(protein=protein, seed=42, noise_level=0.0)
    rng = np.random.default_rng(42)

    # Random sample
    forms = []
    for _ in range(n_samples):
        aa  = rng.choice(AA_LIST)
        sug = rng.choice(SUGAR_LIST)
        sf  = rng.choice(SF_LIST)
        edta = bool(rng.random() > 0.5)
        aa_p = AMINO_ACIDS[aa]; s_p = SUGARS[sug]; sf_p = SURFACTANTS[sf]
        forms.append({
            "aa": aa,
            "aa_conc": float(rng.uniform(aa_p["min"], aa_p["max"])),
            "sugar": sug,
            "sugar_conc": float(rng.uniform(s_p["min"], s_p["max"])),
            "surfactant": sf,
            "surfactant_conc": float(rng.uniform(sf_p["min"], sf_p["max"])),
            "edta": edta,
        })

    X = np.array([formulation_to_vector(f) for f in forms])
    y = np.array([oracle_cont.score(f, add_noise=False) for f in forms])

    # Fit GP to estimate landscape smoothness
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True,
                                   n_restarts_optimizer=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(Xs[:200], y[:200])  # fit on subset for speed

    ls = float(gp.kernel_.length_scale)
    nn_dists = []
    for i in range(min(200, len(Xs))):
        dists = np.linalg.norm(Xs - Xs[i], axis=1)
        dists[i] = np.inf
        nn_dists.append(dists.min())
    mean_nn = float(np.mean(nn_dists))

    best_form, best_score = oracle_cont.global_best_formulation()

    return {
        "protein": protein,
        "n_samples": n_samples,
        "score_mean": float(y.mean()),
        "score_std": float(y.std()),
        "score_max": float(y.max()),
        "score_min": float(y.min()),
        "gp_lengthscale": ls,
        "mean_nn_distance": mean_nn,
        "ls_nn_ratio": ls / mean_nn,
        "landscape_type": "structured" if ls / mean_nn > 1.0 else "rough",
        "best_formulation": best_form,
        "best_score": best_score,
    }


# ── Quick demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", default="mAb_aggregation",
                        choices=list(PROTEIN_PROFILES.keys()))
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--analyse", action="store_true",
                        help="Run landscape analysis (takes ~30s)")
    args = parser.parse_args()

    print(f"Building ExcipientOracle: protein={args.protein}")
    oracle = ExcipientOracle(protein=args.protein, seed=42)

    # Show a few example formulations
    examples = [
        {"aa": "arginine", "aa_conc": 15.0, "sugar": "sucrose",
         "sugar_conc": 70.0, "surfactant": "polysorbate80",
         "surfactant_conc": 0.4, "edta": True},
        {"aa": "methionine", "aa_conc": 1.0, "sugar": "trehalose",
         "sugar_conc": 65.0, "surfactant": "polysorbate80",
         "surfactant_conc": 0.4, "edta": True},
        {"aa": "histidine", "aa_conc": 6.0, "sugar": "mannitol",
         "sugar_conc": 45.0, "surfactant": "poloxamer188",
         "surfactant_conc": 0.5, "edta": False},
        {"aa": "glycine", "aa_conc": 18.0, "sugar": "sucrose",
         "sugar_conc": 70.0, "surfactant": "polysorbate20",
         "surfactant_conc": 0.35, "edta": False},
    ]
    print("\nExample formulations (no noise):")
    for f in examples:
        s = oracle.score(f, add_noise=False)
        print(f"  {f['aa']:12s} {f['aa_conc']:5.1f}mM | "
              f"{f['sugar']:10s} {f['sugar_conc']:5.1f}mM | "
              f"{f['surfactant']:14s} {f['surfactant_conc']:4.2f}% | "
              f"edta={str(f['edta']):<5} → score={s:.3f}")

    best_form, best_score = oracle.global_best_formulation()
    print(f"\nTheoretical best: {best_form}")
    print(f"  Score: {best_score:.4f}")

    print(f"\nBuilding discrete oracle ({args.n_samples} samples)...")
    disc = oracle.make_discrete_oracle(n_samples=args.n_samples, seed=42)
    print(f"  Pool size: {len(disc)}")
    print(f"  global_best: {disc.global_best():.4f}")
    print(f"  global_worst: {disc.global_worst():.4f}")
    print(f"  Score range: {disc._y_raw.min():.3f} – {disc._y_raw.max():.3f}")
    print(f"  Bounds shape: {disc.bounds().shape}")

    if args.analyse:
        print(f"\nAnalysing landscape for {args.protein}...")
        stats = analyse_landscape(args.protein)
        print(f"  Score: mean={stats['score_mean']:.3f} ± {stats['score_std']:.3f}")
        print(f"  GP lengthscale: {stats['gp_lengthscale']:.3f}")
        print(f"  Mean NN distance: {stats['mean_nn_distance']:.3f}")
        print(f"  ls/nn ratio: {stats['ls_nn_ratio']:.3f}  → {stats['landscape_type']}")
        print(f"  Best score: {stats['best_score']:.4f}")
