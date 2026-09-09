"""
excipient_oracle_mo.py — Multi-objective extension of excipient_oracle.py

Extends the single-objective ExcipientOracle to emit three separate objectives
(Tm, kD, viscosity) instead of one collapsed stability score, matching the real
antibody campaign's design (Tagg dropped, no hard viscosity constraint, no
combining Tm/kD, no objective weighting anywhere — per the resolved design tree).

Design decisions carried over from the design interview, restated here so this
file is self-contained and auditable:
  - Three objectives, all soft-optimised: Tm (maximise), kD (maximise),
    viscosity (minimise). No hard constraints, no scalar weights anywhere.
  - Per-objective dose-response difficulty is INDEPENDENTLY tunable (separate
    noise level and separate "peakiness"/roughness per objective), not coupled
    through a single protein_profile scalar as in the single-objective oracle.
    This is required to stress-test the per-objective GP-trust diagnostic
    (C2) — a coupled design would confound noise level with mechanism
    relevance and make it impossible to tell whether the diagnostic responds
    to landscape roughness or to noise.
  - Reuses the same 16D one-hot categorical encoding, the same excipient
    catalogue (5 amino acids, 4 sugars, 3 surfactants, EDTA flag), and the
    same discrete-pool / nearest-neighbour lookup pattern as the
    single-objective oracle, so it plugs directly into the existing
    NNOracle-compatible campaign infrastructure (run_egbo_campaign,
    posthoc_llm_comparison.py, etc.) with only the scoring function replaced.

Efficiency notes (this was asked for explicitly):
  - All three objective scores for a formulation are computed from the SAME
    parsed excipient lookup (no redundant dict lookups per objective).
  - Pool construction vectorises the dose-response Gaussian evaluation across
    all 500 samples at once with numpy broadcasting rather than a per-row
    Python loop (the single-objective oracle's `make_discrete_oracle` builds
    the pool with a Python loop over formulations because it also has to
    stratify by excipient combination; here we keep the same stratified
    sampling for the categorical assignment, since that part is cheap — N=500
    iterations of dict lookups is not a real bottleneck — but the actual
    SCORING pass, which uses to be one oracle.score() call per formulation,
    is vectorised into three numpy array computations over all rows at once).
  - Pareto-front computation (needed downstream for the Pareto-bookkeeping
    layer) is included here using a vectorised non-dominated-sort rather than
    an O(n^2) pure-Python double loop, since with N=500 discrete samples a
    naive O(n^2) implementation in Python is measurably slow (~0.25M
    comparisons) versus a numpy-vectorised version (~5-10x faster in
    practice for this N).

Usage:
    from excipient_oracle_mo import MultiObjectiveExcipientOracle

    oracle = MultiObjectiveExcipientOracle(
        protein="mAb_aggregation",
        tm_difficulty=0.5, kd_difficulty=0.5, viscosity_difficulty=0.5,
        tm_noise=0.15, kd_noise=0.15, viscosity_noise=0.15,
        seed=42,
    )
    disc = oracle.make_discrete_oracle(n_samples=500, seed=42)
    # disc.query_mo(x) -> np.array([tm, kd, viscosity])  (raw units, not normalised)
    # disc.pareto_front() -> indices of non-dominated formulations
    # disc.bounds() -> (16, 2) same as single-objective oracle
"""

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

# Reuse the excipient catalogue and encoding from the single-objective oracle —
# do NOT redefine it here, to guarantee the 16D vector layout stays identical
# and formulation_to_vector/vector_to_formulation remain interchangeable
# between single- and multi-objective code paths.
from excipient_oracle import (
    AMINO_ACIDS, SUGARS, SURFACTANTS,
    AA_LIST, SUGAR_LIST, SF_LIST, FEATURE_DIM,
    formulation_to_vector, vector_to_formulation,
)


# ── Per-objective difficulty profile ────────────────────────────────────────────

@dataclass
class ObjectiveProfile:
    """
    Independently tunable per-objective landscape parameters.

    difficulty  : in [0,1]. Higher = sharper/narrower dose-response peaks
                  (rougher landscape, harder for a GP to fit at small N).
                  Maps to the Gaussian breadth multiplier: breadth_eff =
                  breadth_base * (1.3 - difficulty), so difficulty=0 gives
                  wide/smooth peaks and difficulty=1 gives narrow/sharp peaks.
    noise_frac  : measurement noise as a fraction of the objective's dynamic
                  range (mirrors CI/range from real assay data — see Waibel
                  Table S2: Tm~0.013, kD~0.096, RM_Agi~0.06, after the
                  sign-error correction). This directly controls the
                  ground-truth noise-to-signal ratio the per-objective trust
                  diagnostic must recover.

                  CALIBRATION CAVEAT (2026-09-08 SI re-read): the RM_Agi
                  noise_frac (~0.06) was derived from Waibel Table S2, but
                  this oracle's third objective is VISCOSITY, not RM_Agi —
                  a deliberate substitution onto a more standard formulation
                  axis (see VISC_RANGE below) — so that derivation does NOT
                  transfer to viscosity_noise. Table S4 (the real viscosity
                  data) reports no replicate CI at all (5 repeats averaged,
                  no variance given), so viscosity_noise=0.10 has no real
                  calibration source and is a placeholder, unlike tm_noise
                  and kd_noise which are both exact CI/range ratios from
                  Table S2.
    which_pathways : which of the protein's degradation pathways this
                  objective is mechanistically sensitive to, as a dict of
                  pathway_name -> relative_weight. E.g. Tm cares about
                  denaturation_tendency and aggregation_tendency; kD cares
                  almost entirely about aggregation_tendency; viscosity is
                  driven by total excipient load, not pathway-specific.

    CALIBRATION CAVEAT ON `difficulty` (2026-09-08): breadth/difficulty is
    only qualitatively sanity-checked against real data for 2 of the 16
    encoded dimensions (arginine, sorbitol — Waibel Fig S9 shows broad,
    smooth monotonic-ish trends, not sharp peaks, arguing against a high
    difficulty value there). The other 4 amino acids, 3 remaining sugars,
    and all surfactants are entirely unconstrained by real data — Waibel's
    design never varied them. Treat `difficulty` outside arginine/sorbitol
    as an assumed, not calibrated, landscape shape.
    """
    difficulty: float = 0.5
    noise_frac: float = 0.10
    which_pathways: Dict[str, float] = field(default_factory=dict)


# Real-unit target ranges, loosely anchored to the Waibel dataset so the
# synthetic oracle's dynamic range is realistic for a real mAb formulation
# campaign (not required to match exactly — this is a synthetic oracle for
# stress-testing, not a surrogate of the Waibel data itself).
#
# CORRECTED 2026-09-08 after a full SI re-read (Table S2 = 33 real Tm/kD/
# RM_Agi measurements w/ CIs, Table S4 = real viscosity, n=8 @ 125 mg/mL):
#   TM_RANGE was (60.0, 75.0), span 15°C — real observed span is 63.4-71.3°C
#     (7.9°C), so the old range was ~2x too wide. Since tm_noise=0.013 is a
#     FRACTION of this range (see ObjectiveProfile.noise_frac docstring),
#     the old wider range silently doubled the actual injected noise
#     (0.013*15=0.195°C absolute, vs. the real ±0.1°C the fraction was
#     derived from). Tightened to match the real span instead of
#     re-deriving noise_frac, per Table S2.
#   VISC_RANGE was (2.0, 25.0), span 23 cP — real Table S4 range is
#     3.0-10.6 cP, so the old range was ~2.5x too wide (most of it dead
#     space no real formulation in this system occupies). Tightened to
#     (2, 12) for a small margin beyond the observed range. NOTE:
#     viscosity_noise=0.10 remains uncalibrated (see ObjectiveProfile
#     docstring) — Table S4 gives no replicate CI to derive a real
#     noise_frac from.
#   KD_RANGE was NOT changed: kD's real observed span (-24.4 to 48.6,
#     i.e. 73.0) already closely matches the coded (-25, 50) span of 75 —
#     this one was already correctly anchored.
TM_RANGE   = (63.0, 72.0)     # °C — tightened to match Waibel Table S2's real span
KD_RANGE   = (-25.0, 50.0)    # mL/g  (sign matters — see corrected Waibel data)
VISC_RANGE = (2.0, 12.0)      # cP (lower is better) — tightened to match Table S4


# ── Multi-objective oracle core ─────────────────────────────────────────────────

class MultiObjectiveExcipientOracle:
    """
    Emits (Tm, kD, viscosity) as three separate, independently-noised outputs.
    Same excipient catalogue and 16D encoding as the single-objective oracle;
    only the scoring function differs.
    """

    def __init__(self, protein: str = "mAb_aggregation", seed: int = 42,
                 tm_difficulty: float = 0.5, kd_difficulty: float = 0.5,
                 viscosity_difficulty: float = 0.5,
                 tm_noise: float = 0.10, kd_noise: float = 0.10,
                 viscosity_noise: float = 0.10):
        from excipient_oracle import PROTEIN_PROFILES
        self.protein = PROTEIN_PROFILES[protein]
        self._protein_name = protein
        self.rng = np.random.default_rng(seed)

        # Independently tunable per-objective profiles.
        # which_pathways encodes WHICH degradation mechanism each objective
        # is sensitive to — this is what lets the LLM's mechanism knowledge
        # ("arginine helps aggregation, doesn't help viscosity directly")
        # be genuinely testable per objective, and what lets the crossed
        # wrong-prior stress test extend cleanly to the multi-objective case.
        self.tm_profile = ObjectiveProfile(
            difficulty=tm_difficulty, noise_frac=tm_noise,
            which_pathways={"denaturation_tendency": 0.6,
                            "aggregation_tendency": 0.4},
        )
        self.kd_profile = ObjectiveProfile(
            difficulty=kd_difficulty, noise_frac=kd_noise,
            which_pathways={"aggregation_tendency": 1.0},
        )
        self.visc_profile = ObjectiveProfile(
            difficulty=viscosity_difficulty, noise_frac=viscosity_noise,
            which_pathways={},  # viscosity is load-driven, not pathway-driven
        )

    # ── Vectorised dose-response scoring ────────────────────────────────────────
    # All three objectives are computed together from one parsed formulation
    # batch to avoid redundant catalogue lookups.

    def _score_batch(self, aa_idx, aa_conc, sugar_idx, sugar_conc,
                     sf_idx, sf_conc, edta) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorised scoring over a batch of N formulations.
        Inputs are already-decoded arrays of length N (indices + concentrations).
        Returns (tm, kd, viscosity), each shape (N,), in raw units, NOISE-FREE.
        """
        n = len(aa_idx)

        aa_names  = np.array(AA_LIST)[aa_idx]
        sug_names = np.array(SUGAR_LIST)[sugar_idx]
        sf_names  = np.array(SF_LIST)[sf_idx]

        # Precompute per-excipient optimal concentration + breadth arrays
        aa_opt = np.array([AMINO_ACIDS[a]["optimal_conc"] for a in aa_names])
        aa_bw  = np.array([AMINO_ACIDS[a]["breadth"]      for a in aa_names])
        sug_opt = np.array([SUGARS[s]["optimal_conc"] for s in sug_names])
        sug_bw  = np.array([SUGARS[s]["breadth"]      for s in sug_names])
        sf_opt  = np.array([SURFACTANTS[s]["optimal_conc"] for s in sf_names])
        sf_bw   = np.array([SURFACTANTS[s]["breadth"]      for s in sf_names])

        # Pathway-relevance weight per amino acid (anti_aggregation vs
        # antioxidant vs buffer vs other) — same categorical logic as the
        # single-objective oracle's _aa_contribution, vectorised.
        aa_is_antiagg = np.array(
            ["anti_aggregation" in AMINO_ACIDS[a]["roles"] for a in aa_names])
        aa_is_antiox = np.array(
            ["antioxidant" in AMINO_ACIDS[a]["roles"] for a in aa_names])
        aa_is_buffer = np.array(
            ["buffer" in AMINO_ACIDS[a]["roles"] for a in aa_names])

        def gaussian_peak(x, opt, bw, difficulty):
            eff_bw = bw * (1.3 - difficulty)
            return np.exp(-0.5 * ((x - opt) / eff_bw) ** 2)

        # ── Tm: driven by denaturation_tendency (mostly sugars) + some aggregation ──
        tm_aa_peak = gaussian_peak(aa_conc, aa_opt, aa_bw, self.tm_profile.difficulty)
        tm_aa_contrib = np.where(
            aa_is_antiagg, 0.20 * self.protein.aggregation_tendency * tm_aa_peak,
            np.where(aa_is_buffer, 0.10 * self.protein.denaturation_tendency * tm_aa_peak,
                    0.08 * self.protein.denaturation_tendency * tm_aa_peak))
        tm_sug_peak = gaussian_peak(sugar_conc, sug_opt, sug_bw, self.tm_profile.difficulty)
        tm_sug_contrib = 0.55 * self.protein.denaturation_tendency * tm_sug_peak
        tm_raw = tm_aa_contrib + tm_sug_contrib
        # Fixed ceiling, NOT per-profile (avoids the cancellation bug kD
        # had) — but the ceiling VALUE must be the actual maximum tm_raw
        # achievable across all four protein profiles, not an assumed 1.0.
        # tm_raw = 0.20*agg_contribution + 0.55*denat_contribution (roughly;
        # see _aa_contribution/_score_batch for the exact per-role weights),
        # and no profile can reach agg=1.0 AND denat=1.0 simultaneously —
        # the highest achievable value across the four defined profiles is
        # enzyme_labile's ≈0.4925 (agg=0.40, denat=0.75). Using ceiling=1.0
        # meant even the best-case profile only filled ~49% of the stated
        # Tm range, and mAb_aggregation (the default campaign protein)
        # compressed to just 33.5% (span=5.0°C of the intended 15°C) —
        # verified empirically. This starves Tm of dynamic range and
        # effectively turns 3-objective campaigns into 2-objective ones
        # (kD vs viscosity) with Tm as a near-constant, since a Pareto
        # front built from noise-level Tm variation carries no real
        # trade-off signal on that axis.
        TM_CEILING_MAX_ACHIEVABLE = 0.4925  # = max_profile(0.20*agg + 0.55*denat)
        tm_ceiling = TM_CEILING_MAX_ACHIEVABLE
        tm_frac = np.clip(tm_raw / tm_ceiling, 0.0, 1.0)
        tm = TM_RANGE[0] + tm_frac * (TM_RANGE[1] - TM_RANGE[0])

        # ── kD: driven almost entirely by aggregation_tendency (mostly amino acid) ──
        kd_aa_peak = gaussian_peak(aa_conc, aa_opt, aa_bw, self.kd_profile.difficulty)
        kd_aa_contrib = np.where(
            aa_is_antiagg, 0.85 * self.protein.aggregation_tendency * kd_aa_peak,
            0.05 * self.protein.aggregation_tendency * kd_aa_peak)
        # Surfactant synergy: arginine + polysorbate80/20 boosts kD specifically.
        #
        # CALIBRATION CAVEAT (2026-09-08 SI re-read): Waibel Fig S11 shows
        # real arginine concentration strongly ANTI-correlating with kD
        # (Spearman r=-0.94) for the real molecule studied (bococizumab).
        # That's the opposite direction from this mechanism's assumption.
        # This isn't necessarily a bug -- arginine's effect on colloidal
        # stability is documented as protein-specific (see waibel_mab_data.py
        # in the sibling mo_bo_pipeline project for the same real dataset),
        # and this oracle's synergy mechanism is deliberately modelling a
        # DIFFERENT synthetic protein profile, not bococizumab specifically.
        # But if this mechanism was ever meant to reflect general literature
        # consensus rather than one assumed profile, Waibel's real data is a
        # direct counterexample worth citing, not leaving implicit.
        sf_synergy_partner = np.array(
            [SURFACTANTS[s]["synergy_partner"] for s in sf_names])
        sf_synergy_strength = np.array(
            [SURFACTANTS[s]["synergy_strength"] for s in sf_names])
        is_synergy = (sf_synergy_partner == aa_names)
        sf_peak = gaussian_peak(sf_conc, sf_opt, sf_bw, self.kd_profile.difficulty)
        kd_synergy = np.where(
            is_synergy,
            sf_synergy_strength * 1.5 * self.protein.aggregation_tendency *
            kd_aa_peak * sf_peak, 0.0)
        kd_raw = kd_aa_contrib + kd_synergy
        # Fixed ceiling, NOT per-profile. A per-profile ceiling that scales
        # with aggregation_tendency (as an earlier version of this code did)
        # cancels algebraically against the aggregation_tendency term in
        # kd_raw's numerator, making kd_frac IDENTICAL across all protein
        # profiles regardless of aggregation_tendency — i.e. arginine would
        # always be the best amino acid for kD no matter which protein is
        # being modelled. That erases the entire point of having per-protein
        # mechanism differences and breaks the wrong-prior stress test: if
        # the "correct" excipient never changes with the protein profile,
        # a wrong mechanism prior can never actually be wrong.
        # Using a fixed ceiling of 1.0 means low-aggregation-tendency
        # profiles (e.g. mAb_oxidation, agg=0.25) legitimately have a
        # compressed, muted kD response — which is the physically correct
        # behaviour: if aggregation isn't this protein's problem, no amount
        # of arginine should make a dramatic difference to kD.
        kd_ceiling = 1.0
        kd_frac = np.clip(kd_raw / kd_ceiling, 0.0, 1.0)
        kd = KD_RANGE[0] + kd_frac * (KD_RANGE[1] - KD_RANGE[0])

        # ── Viscosity: driven by TOTAL excipient load, not pathway-specific ──
        # Higher concentration of aa + sugar => higher viscosity (worse).
        # This objective deliberately has NO which_pathways relevance — it's
        # a physical load effect, testing whether the LLM (and the diagnostic)
        # correctly treat it differently from the two mechanism-driven objectives.
        aa_frac  = aa_conc  / np.array([AMINO_ACIDS[a]["max"] for a in aa_names])
        sug_frac = sugar_conc / np.array([SUGARS[s]["max"] for s in sug_names])
        load = 0.5 * aa_frac + 0.5 * sug_frac      # in [0,1]
        # difficulty here controls how sharply viscosity rises with load
        visc_exp = 1.0 + 2.0 * self.visc_profile.difficulty
        visc_raw = load ** (1.0 / visc_exp)         # in [0,1], higher=more viscous
        viscosity = VISC_RANGE[0] + visc_raw * (VISC_RANGE[1] - VISC_RANGE[0])

        return tm, kd, viscosity

    def score(self, formulation: dict, add_noise: bool = True) -> np.ndarray:
        """Single-formulation convenience wrapper around the batch scorer."""
        aa_idx    = np.array([AA_LIST.index(formulation["aa"])])
        aa_conc   = np.array([float(formulation["aa_conc"])])
        sugar_idx = np.array([SUGAR_LIST.index(formulation["sugar"])])
        sugar_conc= np.array([float(formulation["sugar_conc"])])
        sf_idx    = np.array([SF_LIST.index(formulation["surfactant"])])
        sf_conc   = np.array([float(formulation["surfactant_conc"])])
        edta      = np.array([bool(formulation.get("edta", False))])

        tm, kd, visc = self._score_batch(aa_idx, aa_conc, sugar_idx, sugar_conc,
                                         sf_idx, sf_conc, edta)
        out = np.array([tm[0], kd[0], visc[0]])

        if add_noise:
            noise = np.array([
                self.rng.normal(0, self.tm_profile.noise_frac *
                                (TM_RANGE[1]-TM_RANGE[0])),
                self.rng.normal(0, self.kd_profile.noise_frac *
                                (KD_RANGE[1]-KD_RANGE[0])),
                self.rng.normal(0, self.visc_profile.noise_frac *
                                (VISC_RANGE[1]-VISC_RANGE[0])),
            ])
            out = out + noise
            out[0] = np.clip(out[0], *TM_RANGE)
            out[1] = np.clip(out[1], *KD_RANGE)
            out[2] = np.clip(out[2], VISC_RANGE[0], VISC_RANGE[1]*1.5)

        return out

    def make_discrete_oracle(self, n_samples: int = 500,
                              seed: int = 42) -> "DiscreteMOExcipientOracle":
        return DiscreteMOExcipientOracle.build(self, n_samples=n_samples, seed=seed)


# ── Discrete multi-objective oracle (NNOracle-MO-compatible) ──────────────────

class DiscreteMOExcipientOracle:
    """
    Discrete pool version — same stratified-sampling structure as the
    single-objective DiscreteExcipientOracle, but stores a (N,3) objective
    matrix instead of a (N,) score vector, and adds Pareto-front computation.
    """

    def __init__(self, X_raw, Y_raw, forms, scaler):
        self._X_raw = X_raw     # (N, 16)
        self._Y_raw = Y_raw     # (N, 3) — [Tm, kD, viscosity]
        self._forms = forms
        self._scaler = scaler
        self._queried = set()
        self._pareto_idx_cache = None

    @classmethod
    def build(cls, oracle: MultiObjectiveExcipientOracle,
              n_samples: int = 500, seed: int = 42) -> "DiscreteMOExcipientOracle":
        rng = np.random.default_rng(seed)
        forms = []

        # Same stratified sampling pattern as the single-objective oracle:
        # guarantee every (aa, sugar, surfactant, edta) combination is
        # represented before filling the remainder randomly.
        for aa, sugar, sf, edta in itertools.product(
            AA_LIST, SUGAR_LIST, SF_LIST, [False, True]
        ):
            n_conc = max(1, n_samples // 120)
            for _ in range(n_conc):
                aa_p, s_p, sf_p = AMINO_ACIDS[aa], SUGARS[sugar], SURFACTANTS[sf]
                aa_c = round(float(rng.uniform(aa_p["min"], aa_p["max"]))
                            / aa_p["step"]) * aa_p["step"]
                s_c  = round(float(rng.uniform(s_p["min"], s_p["max"]))
                            / s_p["step"]) * s_p["step"]
                sf_c = round(float(rng.uniform(sf_p["min"], sf_p["max"]))
                            / sf_p["step"], 1) * sf_p["step"]
                forms.append({"aa": aa, "aa_conc": aa_c, "sugar": sugar,
                              "sugar_conc": s_c, "surfactant": sf,
                              "surfactant_conc": sf_c, "edta": edta})

        while len(forms) < n_samples:
            aa, sug, sf = rng.choice(AA_LIST), rng.choice(SUGAR_LIST), rng.choice(SF_LIST)
            edta = bool(rng.random() > 0.5)
            aa_p, s_p, sf_p = AMINO_ACIDS[aa], SUGARS[sug], SURFACTANTS[sf]
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

        forms = forms[:n_samples]

        # Vectorised scoring pass over ALL formulations at once — this is the
        # main efficiency win versus calling oracle.score() in a Python loop
        # (500x calls, each re-doing dict lookups + Gaussian evals one row at
        # a time). Decode all formulations into index/concentration arrays
        # first, then call _score_batch ONCE.
        aa_idx    = np.array([AA_LIST.index(f["aa"]) for f in forms])
        aa_conc   = np.array([f["aa_conc"] for f in forms])
        sugar_idx = np.array([SUGAR_LIST.index(f["sugar"]) for f in forms])
        sugar_conc= np.array([f["sugar_conc"] for f in forms])
        sf_idx    = np.array([SF_LIST.index(f["surfactant"]) for f in forms])
        sf_conc   = np.array([f["surfactant_conc"] for f in forms])
        edta      = np.array([f["edta"] for f in forms])

        tm, kd, visc = oracle._score_batch(aa_idx, aa_conc, sugar_idx, sugar_conc,
                                           sf_idx, sf_conc, edta)

        # Vectorised noise addition (one rng draw per objective per row,
        # not per-formulation Python calls)
        n = len(forms)
        tm_noise = oracle.rng.normal(
            0, oracle.tm_profile.noise_frac * (TM_RANGE[1]-TM_RANGE[0]), n)
        kd_noise = oracle.rng.normal(
            0, oracle.kd_profile.noise_frac * (KD_RANGE[1]-KD_RANGE[0]), n)
        visc_noise = oracle.rng.normal(
            0, oracle.visc_profile.noise_frac * (VISC_RANGE[1]-VISC_RANGE[0]), n)

        tm = np.clip(tm + tm_noise, *TM_RANGE)
        kd = np.clip(kd + kd_noise, *KD_RANGE)
        visc = np.clip(visc + visc_noise, VISC_RANGE[0], VISC_RANGE[1] * 1.5)

        Y_raw = np.column_stack([tm, kd, visc])
        X_raw = np.array([formulation_to_vector(f) for f in forms])

        scaler = StandardScaler()
        scaler.fit(X_raw)

        return cls(X_raw, Y_raw, forms, scaler)

    def bounds(self) -> np.ndarray:
        return np.column_stack([np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM)])

    def objective_directions(self) -> List[str]:
        """['max', 'max', 'min'] for Tm, kD, viscosity — used by Pareto logic."""
        return ["max", "max", "min"]

    def objective_names(self) -> List[str]:
        return ["Tm", "kD", "viscosity"]

    def query_mo(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Nearest-neighbour lookup. Returns (objective_vector, pool_index)."""
        x_s = self._scaler.transform(x.reshape(1, -1))[0]
        X_s = self._scaler.transform(self._X_raw)
        unqueried = [i for i in range(len(self._X_raw)) if i not in self._queried]
        if not unqueried:
            unqueried = list(range(len(self._X_raw)))
        dists = np.linalg.norm(X_s[unqueried] - x_s, axis=1)
        chosen = unqueried[int(np.argmin(dists))]
        self._queried.add(chosen)
        return self._Y_raw[chosen].copy(), chosen

    def get_formulation(self, idx: int) -> dict:
        return self._forms[idx]

    def pareto_front(self, indices: List[int] = None) -> np.ndarray:
        """
        Vectorised non-dominated sort over the given indices (default: all
        pool points). Returns the indices (into the ORIGINAL pool, not into
        `indices`) of the non-dominated set.

        Vectorised implementation: for N points and M objectives, builds an
        (N,N) domination matrix via broadcasting rather than an O(N^2) pure
        Python double loop with per-pair comparisons. For N=500 this is the
        difference between ~250,000 Python-level comparisons and a handful
        of numpy array ops — meaningfully faster and the natural choice given
        this will be called every batch during a campaign.
        """
        idx = np.arange(len(self._Y_raw)) if indices is None else np.asarray(indices)
        Y = self._Y_raw[idx].copy()

        # Convert to "all maximise" convention so we can compare uniformly
        directions = self.objective_directions()
        for j, d in enumerate(directions):
            if d == "min":
                Y[:, j] = -Y[:, j]

        n = len(Y)
        # domination[i,k] = True if point k dominates point i
        # k dominates i  <=>  Y[k] >= Y[i] elementwise AND Y[k] > Y[i] somewhere
        ge = (Y[None, :, :] >= Y[:, None, :]).all(axis=2)   # ge[i,k]
        gt = (Y[None, :, :] >  Y[:, None, :]).any(axis=2)   # gt[i,k]
        dominated_by_someone = (ge & gt).any(axis=1)         # shape (n,)

        non_dominated_local = ~dominated_by_someone
        result = idx[non_dominated_local]
        if indices is None:
            self._pareto_idx_cache = result
        return result

    def hypervolume(self, indices: List[int] = None,
                    reference_point: np.ndarray = None) -> float:
        """
        Correct hypervolume via pymoo's exact indicator (union of dominated
        hyperrectangles, not a naive per-point maximum). The earlier version
        of this method returned max(individual point volumes) as a proxy,
        which systematically underestimates true HV whenever more than one
        Pareto point is present (their dominated regions overlap and a
        simple max ignores the union) — verified against pymoo's reference
        implementation to underestimate by ~60% on a typical 27-point front,
        which is too large an error to treat as "monitoring only, direction
        is fine." Used only as a reporting/monitoring metric, still not as
        an acquisition signal, per the earlier design decision that
        hypervolume-driven acquisition is not trusted in this campaign's
        low-N/high-D regime.
        """
        from pymoo.indicators.hv import HV

        pf_idx = self.pareto_front(indices)
        Y = self._Y_raw[pf_idx].copy()
        all_idx = np.arange(len(self._Y_raw)) if indices is None else np.asarray(indices)
        Y_all = self._Y_raw[all_idx].copy()

        directions = self.objective_directions()
        # pymoo's HV assumes minimisation of all objectives — convert
        # maximise objectives to minimise by negation (consistent sign
        # convention used elsewhere in this class for Pareto comparisons).
        for j, d in enumerate(directions):
            if d == "max":
                Y[:, j] = -Y[:, j]
                Y_all[:, j] = -Y_all[:, j]

        if reference_point is None:
            ref = Y_all.max(axis=0) + 0.1 * (Y_all.max(axis=0) - Y_all.min(axis=0) + 1e-9)
        else:
            ref = np.asarray(reference_point, dtype=float).copy()
            for j, d in enumerate(directions):
                if d == "max":
                    ref[j] = -ref[j]

        hv_calc = HV(ref_point=ref)
        return float(hv_calc(Y))

    def __len__(self):
        return len(self._X_raw)


# ── Quick self-test / demo ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("Building multi-objective excipient oracle (mAb_aggregation)...")
    t0 = time.time()
    oracle = MultiObjectiveExcipientOracle(
        protein="mAb_aggregation",
        tm_difficulty=0.5, kd_difficulty=0.7, viscosity_difficulty=0.3,
        tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10,   # Waibel-anchored
        seed=42,
    )
    disc = oracle.make_discrete_oracle(n_samples=500, seed=42)
    t1 = time.time()
    print(f"  Built in {t1-t0:.3f}s  (N={len(disc)})")

    print(f"\n  Objectives: {disc.objective_names()}  "
          f"directions: {disc.objective_directions()}")
    print(f"  Tm  range: [{disc._Y_raw[:,0].min():.1f}, {disc._Y_raw[:,0].max():.1f}] °C")
    print(f"  kD  range: [{disc._Y_raw[:,1].min():.1f}, {disc._Y_raw[:,1].max():.1f}] mL/g")
    print(f"  Visc range: [{disc._Y_raw[:,2].min():.1f}, {disc._Y_raw[:,2].max():.1f}] cP")

    t2 = time.time()
    pf = disc.pareto_front()
    t3 = time.time()
    print(f"\n  Pareto front: {len(pf)} / {len(disc)} formulations "
          f"(computed in {t3-t2:.4f}s)")

    print(f"\n  Example Pareto-optimal formulations:")
    for i in pf[:5]:
        f = disc.get_formulation(i)
        tm, kd, visc = disc._Y_raw[i]
        print(f"    {f['aa']:10s} {f['aa_conc']:5.1f}mM | {f['sugar']:10s} "
              f"{f['sugar_conc']:5.1f}mM | {f['surfactant']:14s} "
              f"{f['surfactant_conc']:4.2f}%  ->  Tm={tm:.1f} kD={kd:+.1f} visc={visc:.1f}")

    hv = disc.hypervolume()
    print(f"\n  Hypervolume (monitoring metric only, not acquisition): {hv:.2f}")

    print(f"\nTotal build+analysis time: {time.time()-t0:.3f}s")
