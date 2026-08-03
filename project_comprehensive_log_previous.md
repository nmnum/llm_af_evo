# Comprehensive Project Log: Self-Driving Lab — Adaptive Campaign Planner for LLM-BO

**Project**: Simulating experimental campaigns to understand which optimisation strategies work, fail, or break under different biological and experimental regimes, and deriving design rules for an adaptive campaign planner.

**Author**: Neha Mungale
**Supervisor**: David Shorthouse (UCL School of Pharmacy)
**Date compiled**: 2026-08-01
**Scope**: Everything — from the original cancer drug screening work through the pharmaceutical formulation pivot, single-objective stress tests, multi-objective oracle construction, bug diagnosis and fixing, and the final wrong-prior findings.

---

## Table of Contents

1. [Project Goal and Motivation](#1-project-goal-and-motivation)
2. [Phase 0: Cancer Drug Screening (Pre-Pivot)](#2-phase-0-cancer-drug-screening-pre-pivot)
3. [The Pivot to Pharmaceutical Formulation](#3-the-pivot-to-pharmaceutical-formulation)
4. [Phase 1: Single-Objective Excipient Oracle](#4-phase-1-single-objective-excipient-oracle)
5. [Phase 1.5: Synthetic Benchmark Null Result](#5-phase-15-synthetic-benchmark-null-result)
6. [Phase 2: Single-Objective Stress Test](#6-phase-2-single-objective-stress-test)
7. [Phase 3: Waibel Dataset Extraction](#7-phase-3-waibel-dataset-extraction)
8. [Phase 4: Multi-Objective Oracle Construction](#8-phase-4-multi-objective-oracle-construction)
9. [Phase 5: Multi-Objective Campaign Runner](#9-phase-5-multi-objective-campaign-runner)
10. [Phase 6: Bug Discovery and Fixing](#10-phase-6-bug-discovery-and-fixing)
11. [Phase 7: Multi-Objective Campaign Results](#11-phase-7-multi-objective-campaign-results)
12. [Phase 8: Wrong-Prior Stress Test in Multi-Objective](#12-phase-8-wrong-prior-stress-test-in-multi-objective)
13. [The Four Claims — Status Assessment](#13-the-four-claims--status-assessment)
14. [Literature Context](#14-literature-context)
15. [Future Directions: RL Meta-Controller](#15-future-directions-rl-meta-controller)
16. [Complete File Inventory](#16-complete-file-inventory)
17. [Key Numbers Reference](#17-key-numbers-reference)

---

## 1. Project Goal and Motivation

### The core question

When a self-driving laboratory runs an experimental campaign, it must decide **which optimisation strategy to use at each step**. Should it trust its Gaussian process surrogate model and exploit? Should it explore randomly? Should it ask a large language model (LLM) for proposals based on domain knowledge? These decisions are currently made heuristically — by a human researcher using intuition, or by a fixed algorithm that never adapts.

This project asks: **can we systematise and automate these strategy-level decisions?**

### Why this matters

Most Bayesian optimisation (BO) research focuses on improving acquisition functions — the mathematical rule that decides where to sample next. But in real experimental campaigns, the bigger question is often *which type of strategy to deploy at all*: a data-driven GP-based method, a knowledge-driven LLM-based method, or a simple random search. This is a meta-level decision that current systems make once (at design time) and never revisit.

The project's thesis is that the optimal strategy changes over the lifetime of a campaign, depending on how much data has been collected, how trustworthy the surrogate model is for each objective, and what domain knowledge is available. An adaptive planner that detects these conditions and switches strategies accordingly should outperform any fixed strategy.

### What the project does NOT do

It does not propose new optimisation algorithms. It does not improve GP kernels or acquisition functions. It systematises the strategy-selection decision that is currently made heuristically.

---

## 2. Phase 0: Cancer Drug Screening (Pre-Pivot)

### What was done

The project originally focused on multi-objective Bayesian optimisation for cancer drug screening using GDSC (Genomics of Drug Sensitivity in Cancer) data, following the Aqeeli et al. framework [1].

- **Dataset**: GDSC CRC5 — 5 colorectal cancer cell lines (SNU-C1, LS-1034, LS-513, LS-123, NCI-H747), 349 drugs, with IC50 per cell line as a separate objective
- **Domain shift experiments**: Tested whether BO recovery differs when training and target distributions differ in difficulty level (e.g., KRAS-mutant → BRAF-mutant transfer)
- **Results**: BO recovery was 80–100% across all shift levels; strategy_gap near zero or negative (greedy matches or beats BO within-subtype)
- **Regime analysis**: Varied batch size (3, 8) and training size (5, 10, 20, 46) across question difficulty levels
- **LoCLO (Low-Confidence Local) regime**: Identified regions where the GP is untrustworthy and local search might help

### Why the pivot was necessary

The cancer drug screening problem has **limited structure for LLM priors**. Drug response (IC50) is essentially a black-box endpoint — there is no mechanistic knowledge that an LLM can inject to predict which drug will work on which cell line. The LLM cannot reason "this cell line has high aggregation tendency, so use arginine" because drug response doesn't have that kind of mechanistic structure.

This meant the project's core question — "when does LLM domain knowledge help BO?" — could not be tested on cancer data. The pivot to pharmaceutical formulation was motivated by three factors:

1. **LLM priors have genuine mechanistic content in formulation**: LLMs know that arginine stabilises against aggregation, that low pH promotes conformational stability, that surfactants prevent interfacial stress. This is real, testable domain knowledge.
2. **Multi-objective with heterogeneous noise**: Real formulation campaigns measure Tm (high precision), kD (low precision), and viscosity (medium precision) simultaneously, creating a natural per-objective GP-trust spectrum.
3. **Self-driving lab alignment**: Pharmaceutical formulation is the canonical SDL use case, already demonstrated by the Shorthouse group [1, 2] and others.

### Artifacts from this phase

- `domain_shift_summary.csv`, `diff_domain_shift_summary.csv` — domain shift results
- `regime_q3_n5/n10/n20.json`, `regime_q8_n5/n10/n20.json`, `regime_summary.csv` — regime analysis
- `loclo_results_batch3/8.json`, `loclo_results_q8_v3.json` — LoCLO analysis
- `fig1_strategy_gap_heatmap.png` through `fig5_conformal_coverage.png` — figures
- `gdsc_poc_loclo_notebook_v2.ipynb` — proof-of-concept notebook

---

## 3. The Pivot to Pharmaceutical Formulation

### The new framing

The project reframed around four distinct claims (C1–C4) that map to testable hypotheses:

| Claim | Statement | What it tests |
|-------|-----------|---------------|
| **C1** | Mechanism-level LLM prior knowledge accelerates search in low-trust, small-N, high-D regimes | Does domain knowledge help when the GP can't be trusted? |
| **C2** | A per-objective GP-trust diagnostic correctly predicts which objectives the GP models well vs poorly | Can we detect GP unreliability per objective? |
| **C3** | A trust-driven mixing weight between LLM and GP strategies adapts sensibly across trust regimes | Does adaptive strategy selection work? |
| **C4** | The end-to-end pipeline (Pareto bookkeeping, structured tags, LLM proposals) works on real multi-objective data | Is the system practical? |

### Why these four claims

The claims form a logical chain: C1 establishes that LLM knowledge helps (the motivation), C2 provides the diagnostic signal needed to detect *when* it helps, C3 uses that signal to adaptively switch strategies, and C4 validates the whole system on real data.

---

## 4. Phase 1: Single-Objective Excipient Oracle

### What was built

A synthetic pharmaceutical formulation oracle (`excipient_oracle.py`) that models protein stability as a function of excipient composition. The oracle is the "ground truth" that the optimisation strategies try to learn — it simulates what a real lab assay would return for a given formulation.

### Design principles

The oracle was designed with specific properties to make the stress test meaningful:

1. **Mechanistic structure**: Each excipient has a dose-response curve tied to specific degradation pathways. Arginine helps aggregation-prone proteins. Methionine helps oxidation-prone proteins. Sugars help thermal denaturation. This means the "correct" excipient depends on the protein profile — which is what makes wrong-prior testing possible.

2. **Synergistic interactions**: Arginine + polysorbate80 have a specific synergy for kD. Sucrose + methionine have a synergy for oxidation protection. These interactions create non-trivial landscape structure that a GP must learn.

3. **Flat regions for wrong excipients**: If you use methionine on an aggregation-prone protein, it doesn't hurt — it just doesn't help. The score is flat, not negative. This is realistic: wrong excipients are wasted budget, not actively harmful.

4. **Sharp optima for synergy combinations**: The arginine+PS80 synergy creates a sharp peak that is hard for a GP to find from sparse data, creating the "low-trust" regime where LLM knowledge should help.

5. **Viscosity penalty**: Higher total excipient concentration increases viscosity, creating a genuine trade-off between stability and manufacturability.

### Excipient catalogue

The formulation space consists of:

| Category | Options | Concentration range |
|----------|---------|-------------------|
| Amino acids (pick 1) | arginine, proline, glycine, methionine, histidine | 0.1–25 mM (varies per amino acid) |
| Sugars (pick 1) | sucrose, trehalose, sorbitol, mannitol | 1–100 mM (varies per sugar) |
| Surfactants (pick 1) | polysorbate80, polysorbate20, poloxamer188 | 0.1–1.0% |
| EDTA | yes/no | 0.01% if yes |

Each excipient has an optimal concentration and a Gaussian "breadth" parameter controlling how sharply the dose-response peaks. For example, arginine peaks at 15 mM with breadth 8.0 (broad), while methionine peaks at 1.0 mM with breadth 0.8 (narrow — you need to get the concentration right).

### 16D feature encoding

Each formulation is encoded as a 16-dimensional vector:

```
[0-4]   Amino acid one-hot (5 values: arginine, proline, glycine, methionine, histidine)
[5]     Amino acid concentration (normalised 0-1)
[6-9]   Sugar one-hot (4 values: sucrose, trehalose, sorbitol, mannitol)
[10]    Sugar concentration (normalised 0-1)
[11-13] Surfactant one-hot (3 values: PS80, PS20, poloxamer188)
[14]    Surfactant concentration (normalised 0-1)
[15]    EDTA flag (0 or 1)
```

This encoding is compatible with the existing NNOracle infrastructure used for the cancer drug screening work, allowing the same campaign runner to be reused.

### Protein profiles

Four protein profiles, each with different degradation pathway tendencies:

| Profile | aggregation_tendency | oxidation_tendency | denaturation_tendency | metal_sensitivity |
|---------|---------------------|-------------------|----------------------|-------------------|
| mAb_aggregation | 0.85 | 0.20 | 0.30 | 0.15 |
| mAb_oxidation | 0.25 | 0.80 | 0.25 | 0.60 |
| enzyme_labile | 0.40 | 0.30 | 0.75 | 0.20 |
| mixed | 0.55 | 0.45 | 0.45 | 0.35 |

The protein profile is fixed at oracle construction and **unknown to the optimiser**. This is the key challenge: the LLM must infer which degradation pathway dominates from early observations and adjust its formulation strategy. The profile determines which excipients are effective — arginine helps aggregation-prone proteins, methionine helps oxidation-prone proteins, sugars help denaturation-prone proteins.

### Scoring formula (single-objective)

The single-objective oracle collapses all pathways into one stability score in [0, 1]:

```
score = w_agg * aggregation_score + w_oxid * oxidation_score + 
        w_denat * denaturation_score - viscosity_penalty
```

where the weights are determined by the protein profile's tendencies. For mAb_aggregation, the aggregation pathway dominates (w_agg is highest), so arginine is the most important excipient. For mAb_oxidation, the oxidation pathway dominates, so methionine is most important.

### Discrete oracle (pool of 500)

For fair comparison between strategies, the oracle creates a discrete pool of 500 formulations sampled via Latin Hypercube Sampling. Each strategy proposes formulations, which are snapped to the nearest pool point. This ensures all strategies see the same set of possible experiments.

---

## 5. Phase 1.5: Synthetic Benchmark Null Result

### What was tested

Before testing on the excipient oracle, the LLM-BO framework was tested on three synthetic/ADA datasets to establish a baseline:

| Dataset | Dimensions | Description |
|---------|-----------|-------------|
| pareto_20210112 | varies | ADA thin-film coatings dataset |
| hartmann6 | 6 | Classic synthetic test function |
| coatings | varies | ADA coatings dataset |

Three conditions were compared:
- **default**: Standard GP-UCB Bayesian optimisation
- **llm_tuned**: GP with LLM-suggested hyperparameter priors
- **oracle**: GP with oracle-provided hyperparameters (upper bound on what tuning can achieve)

### Results

| Dataset | Default AUC | LLM AUC | Oracle AUC |
|---------|------------|---------|------------|
| pareto_20210112 | 0.707 | 0.708 | 0.729 |
| hartmann6 | 0.856 | 0.841 | 0.833 |
| coatings | 0.919 | 0.917 | 0.916 |

### Interpretation

This is the **expected null result**: on smooth synthetic problems where the GP surrogate is already well-specified and trustworthy, there is no room for LLM knowledge to help. The LLM's hyperparameter suggestions don't improve the GP because the GP is already working well. This is important because it establishes that the LLM-BO framework doesn't artificially inflate performance — it only helps when the GP is genuinely struggling.

Saved as `oracle_gap_null_result.json`.

---

## 6. Phase 2: Single-Objective Stress Test

### Design

The stress test is the core experiment for C1. It tests whether LLM domain knowledge genuinely helps optimisation, or whether the LLM is just pattern-matching to common formulations. The design uses **four prior levels** crossed with **four protein profiles**:

| Prior level | Description | What it tests |
|------------|-------------|---------------|
| L1 (mechanism) | Correct mechanism-level prior: "This protein is aggregation-prone. Arginine prevents aggregation via charge-shielding." | Does correct domain knowledge help? |
| L4 (blank) | No domain knowledge: "You are optimising a protein formulation. No further domain knowledge is provided." | Is the mechanism knowledge the active ingredient, or does the LLM help for other reasons? |
| WRONG | Wrong mechanism: aggregation prior on oxidation-prone protein, and vice versa | Does the LLM reason about which pathway to target, or does it propose generic formulations regardless? |
| EGBO | Pure GP-based evolutionary BO (no LLM) | Baseline for data-driven search |
| random | Uniform random search | Lower bound |

### The prompt leakage fix

An earlier version of the LLM prompts contained **explicit concentration recommendations** (e.g., "Global optimum is near: methionine ~1 mM, sucrose ~70 mM, polysorbate80 ~0.4%, EDTA=True"). This was equivalent to telling the LLM where the optimum is — not testing domain knowledge, but testing whether the LLM can follow instructions. The LLM converged to 90% of best at step 10 (first call) on every seed with zero variance — deterministic, not search behaviour.

**Fix**: Stripped all concentration hints from prior texts. L1 now contains only mechanism-level knowledge: "Arginine prevents aggregation via charge-shielding. Polysorbate80 synergises with arginine for kD." No concentrations, no optimal values.

### Results (AUC, mean ± SD, 20 seeds)

**mAb_aggregation** (rough landscape, aggregation pathway dominates):

| Condition | AUC | vs EGBO | vs random |
|-----------|-----|---------|-----------|
| L1 (mechanism) | 0.863 ± 0.048 | +0.044* | +0.113* |
| L4 (blank) | 0.837 ± 0.071 | +0.019 | +0.088* |
| EGBO | 0.818 ± 0.083 | — | +0.069* |
| random | 0.749 ± 0.106 | -0.069* | — |
| WRONG | 0.708 ± 0.140 | -0.110* | -0.041 |

**mAb_oxidation** (smoother landscape, oxidation pathway dominates):

| Condition | AUC | vs EGBO | vs random |
|-----------|-----|---------|-----------|
| L1 (mechanism) | 0.917 ± 0.047 | +0.016 | +0.056* |
| L4 (blank) | 0.889 ± 0.088 | -0.011 | +0.029 |
| EGBO | 0.900 ± 0.069 | — | +0.040* |
| random | 0.860 ± 0.090 | -0.040* | — |
| WRONG | 0.817 ± 0.131 | -0.083* | -0.043* |

(* = p < 0.05, paired t-test)

### Interpretation against the four criteria

| Criterion | mAb_aggregation | mAb_oxidation | Verdict |
|-----------|----------------|---------------|---------|
| L1 > EGBO* | +0.044, p<0.05 ✓ | +0.016, p>0.05 ✗ | Partial |
| L4 ≈ EGBO | +0.019, p=NS ✓ | -0.011, p=NS ✓ | Met |
| WRONG < EGBO* | -0.110, p<0.05 ✓ | -0.083, p<0.05 ✓ | **Met decisively** |
| L1 >> L4 | 0.863 vs 0.837 ✓ | 0.917 vs 0.889 ✓ | Met directionally |

### Key findings

1. **The wrong-prior result is decisive**: On both profiles, giving the LLM the wrong mechanism knowledge makes it perform significantly worse than EGBO and even worse than random search. This rules out the hypothesis that the LLM is outputting generic pharmaceutical formulations regardless of instructions. The LLM is reasoning about which degradation pathway to target, and when that reasoning is wrong, it proposes the wrong class of excipients and actively loses.

2. **The knowledge gradient is clean on mAb_aggregation**: WRONG (0.708) < random (0.749) < EGBO (0.818) ≈ L4-blank (0.837) < L1-mechanism (0.863). This is exactly what the domain-knowledge hypothesis predicts.

3. **L1 does not significantly beat EGBO on mAb_oxidation**: The oxidation landscape is easier (EGBO already reaches 0.900) and the mechanism knowledge is simpler ("use methionine for oxidation"), so there's less room for the LLM to add value.

4. **Blank prior ≈ EGBO**: The LLM without domain knowledge performs similarly to pure GP. This confirms that the mechanism knowledge is the active ingredient, not the LLM's architecture or output format.

### What this validated

- **C1 (single-objective)**: Mechanism-level prior beats GP-only on rough landscapes; wrong prior hurts; blank ≈ EGBO. Validated.
- The single-objective oracle's collapsed score is dominated by the correct pathway, making "wrong mechanism" genuinely wrong.

---

## 7. Phase 3: Waibel Dataset Extraction

### Source

Waibel et al. (2025) "Bayesian Optimization for Efficient Multiobjective Formulation Development of Biologics" [3]. This is a real experimental dataset from ETH Zurich / Novo Nordisk, optimising a monoclonal antibody (bococizumab-IgG1) formulation.

### What was extracted

The Supporting Information (SI) PDF contained 33 formulations with 7 varying inputs and 3 objectives:

| Input | Range | Role |
|-------|-------|------|
| pH | 4.7–7.8 | Affects both Tm and kD (opposing effects) |
| Histidine | 10 mM (fixed) | Buffer |
| Sorbitol | 0–517 mM | Tonicity/stability |
| Arginine | 0–201 mM | Anti-aggregation |
| Aspartic acid | 0–55 mM | pH adjustment |
| Glutamic acid | 0–76 mM | pH adjustment |
| Acetic acid | 0–100 mM | pH adjustment |
| HCl | 0–121 mM | pH adjustment |

| Objective | Range | CI (mean) | Noise-to-signal ratio |
|-----------|-------|-----------|----------------------|
| Tm (melting temperature) | 63.4–71.3 °C | ±0.09 °C | 0.012 (HIGH trust) |
| kD (diffusion interaction parameter) | -24.4 to +48.6 mL/g | ±7.0 mL/g | 0.096 (LOW trust) |
| RM_Agi (retained monomer after agitation) | 1.8–101.1% | ±6.1% | 0.061 (MEDIUM trust) |

### Why this dataset matters

1. **Built-in per-objective noise spectrum**: Tm has very low noise (CI/range = 0.012), kD has high noise (0.096), RM_Agi has medium noise (0.061). This is exactly the per-objective GP-trust spectrum that C2 requires — the diagnostic should assign HIGH trust to Tm, LOW trust to kD, MEDIUM trust to RM_Agi.

2. **Real experimental structure**: 15 initial DoE + 18 BO-guided = 33 total experiments. The campaign structure matches what a real self-driving lab would do.

3. **5 Pareto-optimal formulations**: #14 (Tm=67.9, kD=23.3, RM=101.1), #24 (71.3, 28.5, 95.7), #25 (70.9, 31.5, 100.1), #30 (70.7, 43.7, 96.7), #31 (70.9, 48.6, 96.4).

4. **Objective correlations**: Tm-kD = +0.591 (moderate positive), Tm-RM_Agi = +0.290 (weak), kD-RM_Agi = +0.262 (weak). The Tm-kD correlation means there's a genuine trade-off — you can't always maximise both simultaneously.

### kD sign correction

The original SI had kD values that appeared to be all positive. After checking against the paper text and the physical meaning of kD (negative = attractive protein-protein interactions = bad; positive = repulsive = good), the data was corrected: 16/33 values are negative, 17/33 positive, range [-24.4, +48.6]. This correction is important because the oracle's kD scoring must produce both positive and negative values to be realistic.

Saved to `waibel_mAb_formulation_dataset_CORRECTED.csv`.

### Oracle noise calibration

The multi-objective oracle's noise_frac parameters were calibrated to match the Waibel noise-to-signal ratios:

| Objective | Oracle noise_frac | Waibel CI/range |
|-----------|------------------|-----------------|
| Tm | 0.013 | 0.012 |
| kD | 0.096 | 0.096 |
| viscosity | 0.100 | (estimated from typical formulation viscosity CV) |

This ensures the oracle's noise structure is realistic, not arbitrary.

---

## 8. Phase 4: Multi-Objective Oracle Construction

### What was built

The single-objective oracle was extended to emit three separate objectives (Tm, kD, viscosity) instead of one collapsed score. This is the `excipient_oracle_mo.py` file.

### Key design decisions

1. **Three independent objectives, all soft-optimised**: Tm (maximise), kD (maximise), viscosity (minimise). No hard constraints, no scalar weights. This matches the real Waibel campaign design.

2. **Per-objective difficulty is independently tunable**: Each objective has its own `difficulty` parameter (controlling dose-response peakiness) and `noise_frac` parameter (controlling measurement noise). This is critical for C2 — a coupled design would confound noise level with mechanism relevance and make it impossible to tell whether the diagnostic responds to landscape roughness or to noise.

3. **Same 16D encoding and excipient catalogue**: The multi-objective oracle reuses the same input space as the single-objective version, so it plugs directly into the existing campaign infrastructure.

### Scoring formulas

Each objective is computed from the same parsed formulation but through different mechanistic pathways:

#### Tm (melting temperature)

Tm is driven by **denaturation tendency** (primarily sugars) with a smaller contribution from **aggregation tendency** (anti-aggregation amino acids):

```python
# Amino acid contribution to Tm
if aa_is_anti_agg:
    tm_aa_contrib = 0.20 * aggregation_tendency * aa_peak
elif aa_is_buffer:  # histidine
    tm_aa_contrib = 0.10 * denaturation_tendency * aa_peak
else:
    tm_aa_contrib = 0.08 * denaturation_tendency * aa_peak

# Sugar contribution to Tm (dominant for denaturation-prone proteins)
tm_sug_contrib = 0.55 * denaturation_tendency * sug_peak

tm_raw = tm_aa_contrib + tm_sug_contrib
```

The ceiling for Tm normalisation is `TM_CEILING_MAX_ACHIEVABLE = 0.4925`, computed as the maximum achievable `tm_raw` across all protein profiles (enzyme_labile: agg=0.40, denat=0.75 → 0.20×0.40 + 0.55×0.75 = 0.4925). This ensures the profile with the highest denaturation tendency fills the full Tm range [60, 75]°C, while lower-tendency profiles have compressed Tm response.

**Per-profile Tm range filling** (with ceiling=0.4925):

| Profile | agg | denat | Tm max | Tm fill % |
|---------|-----|-------|--------|-----------|
| mAb_aggregation | 0.85 | 0.30 | 70.2°C | 68.0% |
| mAb_oxidation | 0.25 | 0.25 | 65.7°C | 38.1% |
| enzyme_labile | 0.40 | 0.75 | 75.0°C | 100.0% |
| mixed | 0.55 | 0.45 | 70.9°C | 72.6% |

#### kD (diffusion interaction parameter)

kD is driven almost entirely by **aggregation tendency** (primarily anti-aggregation amino acids like arginine):

```python
# Anti-aggregation amino acid contribution (arginine, proline)
if aa_is_anti_agg:
    kd_aa_contrib = 0.85 * aggregation_tendency * aa_peak
else:
    kd_aa_contrib = 0.05 * aggregation_tendency * aa_peak

# Surfactant synergy: arginine + PS80/PS20 boosts kD
if is_synergy:  # surfactant's synergy_partner == amino acid name
    kd_synergy = synergy_strength * 1.5 * aggregation_tendency * aa_peak * sf_peak
else:
    kd_synergy = 0.0

kd_raw = kd_aa_contrib + kd_synergy
```

The ceiling for kD is fixed at 1.0 (not per-profile). This is critical — a per-profile ceiling that scales with aggregation_tendency would cancel algebraically against the aggregation_tendency term in the numerator, making kD identical across all profiles. The fixed ceiling means low-aggregation profiles legitimately have compressed kD response.

**Per-profile kD range filling** (with ceiling=1.0, arginine + PS80 at optimal concentration):

| Profile | agg | kD max (arginine) | kD fill % | kD max (methionine) |
|---------|-----|-------------------|-----------|---------------------|
| mAb_aggregation | 0.85 | +50.0 mL/g | 100.0% | -21.8 mL/g |
| mAb_oxidation | 0.25 | +13.4 mL/g | 51.2% | -24.1 mL/g |
| enzyme_labile | 0.40 | +36.5 mL/g | 82.0% | -23.5 mL/g |
| mixed | 0.55 | +50.0 mL/g | 100.0% | -22.9 mL/g |

The kD range is [-25, +50] mL/g. The difference between arginine and methionine on mAb_oxidation is 37.5 mL/g — enormous, even on a protein where aggregation is not the dominant pathway. This becomes critically important in the wrong-prior analysis (Section 12).

#### Viscosity

Viscosity is driven by **total excipient load**, not by any specific pathway:

```python
aa_frac = aa_conc / aa_max
sug_frac = sugar_conc / sugar_max
load = 0.5 * aa_frac + 0.5 * sug_frac  # in [0,1]
visc_raw = load ** (1.0 / visc_exp)    # higher = more viscous
```

Viscosity has **no pathway-specific relevance** — it's a physical load effect. This is deliberate: it tests whether the LLM (and the trust diagnostic) correctly treat it differently from the two mechanism-driven objectives.

### Pareto front structure

The oracle computes Pareto fronts using a vectorised non-dominated sort. On the default mAb_aggregation profile, the Pareto front has 27–52 points out of 500 (5–10%), with the following trade-off structure:

| Correlation | Value | Interpretation |
|-------------|-------|---------------|
| Tm-kD | +0.495 | Moderate positive — higher Tm tends to come with higher kD |
| Tm-visc | +0.841 | Strong positive — better Tm costs more viscosity |
| kD-visc | +0.730 | Strong positive — better kD also costs viscosity |

The strong Tm-visc and kD-visc correlations confirm that viscosity is the key trade-off: you can improve stability (Tm, kD) but it costs manufacturability (viscosity).

---

## 9. Phase 5: Multi-Objective Campaign Runner

### What was built

The `excipient_campaign_mo.py` file implements the multi-objective campaign runner with four strategies, a per-objective trust diagnostic, a mixing weight mechanism, and Pareto bookkeeping.

### Strategies

#### 1. mo_random (baseline)

Generates `batch_size * 6` uniform random formulations in the bounded 16D space. Simple but provides a lower bound on performance.

#### 2. mo_egbo (lightweight EGBO)

A lightweight evolutionary Gaussian-process BO strategy:
- Fits a GP per objective on observed data
- Generates a candidate pool of 60 points: half from perturbation around the current Pareto front (exploitation), half random (exploration)
- Computes UCB (upper confidence bound) predictions per objective for each candidate
- Selects the batch by Pareto-ranking the UCB predictions

**Problem identified**: At low n (10–30) in d=16, the GP's UCB predictions are noise-dominated. The Pareto front of UCB predictions selects only 2/60 candidates (extremely selective), and the selected points cluster in the high-Tm/high-kD region without exploring the viscosity trade-off. This makes mo_egbo underperform random at low n.

#### 3. mo_egbo_real (BoTorch qLogNEHVI)

A genuine multi-objective BO strategy using BoTorch's `qLogNoisyExpectedHypervolumeImprovement` acquisition function with pymoo's UNSGA3 evolutionary algorithm:

- Fits per-objective `SingleTaskGP` models with `Standardize` transform, combined into a `ModelListGP`
- Uses `qLogNoisyExpectedHypervolumeImprovement` with `SobolQMCNormalSampler(sample_shape=8)` for acquisition
- Reference point from the full oracle pool (consistent with the campaign's fixed reference)
- `optimize_acqf` with `num_restarts=2, raw_samples=16, maxiter=20`
- Also generates evolutionary candidates via UNSGA3 with energy reference directions
- Combines qLogNEHVI candidates with evolutionary candidates, selects top-batch by acquisition value
- Falls back to lightweight `strategy_mo_egbo` on any failure (with warning logged)

**C++ extension issue**: BoTorch's fused qLogEHVI C++ extension fails to compile because `Python.h` is not found (missing `python3.12-dev` package). It falls back to pure-Python implementation (~3x slower but functional). This is cosmetic — the results are correct, just slower.

#### 4. mo_llm (LLM-guided with trust-weighted mixing)

The LLM strategy is the most complex. It:

1. Computes per-objective trust scores using the trust diagnostic
2. Computes a mixing weight from the trust scores
3. Asks the LLM to propose named formulations (with prior text describing the protein's mechanism)
4. Filters LLM proposals for diversity (within-batch and cross-batch)
5. Combines LLM proposals with GP-generated candidates
6. Selects the final batch using trust-weighted selection probability

### The LLM prompt architecture

The LLM receives:
- The prior text (mechanism-level, blank, or wrong — see below)
- The current best formulations and their measured objectives
- A request to propose new formulations targeting specific trade-offs

The LLM outputs structured JSON with:
- Formulation details (amino acid, concentration, sugar, concentration, surfactant, concentration, EDTA)
- Target objectives (which of Tm/kD/viscosity this proposal is trying to improve)
- Trade-off note (what the proposal sacrifices)

### Prior texts

Three prior texts are used in the multi-objective campaigns:

**agg_L1** (correct for aggregation-prone proteins):
> "You are optimising a protein formulation for an aggregation-prone monoclonal antibody... Arginine: prevents aggregation via charge-shielding. Primarily affects kD; has a smaller effect on Tm. Polysorbate 80: protects against interface aggregation; synergises with arginine specifically for kD. Methionine, EDTA: address oxidation, not this protein's dominant pathway."

**oxid_L1** (correct for oxidation-prone proteins):
> "You are optimising a protein formulation for an oxidation-prone monoclonal antibody... Methionine: sacrificial antioxidant for oxidation-prone proteins. EDTA: chelates metal ions catalysing oxidation. Arginine and polysorbate80's aggregation-specific synergy is LESS relevant here since aggregation is not this protein's dominant degradation pathway."

**blank** (no domain knowledge):
> "You are optimising a protein formulation. You are tracking THREE separate objectives simultaneously: Tm (higher=better), kD (higher=better), viscosity (lower=better). No further domain knowledge is provided; explore the space systematically."

The prior mapping is:
- `--prior_level L1` on `mAb_aggregation` → uses `agg_L1` (correct)
- `--prior_level L1` on `mAb_oxidation` → uses `oxid_L1` (correct)
- `--prior_level WRONG` on `mAb_oxidation` → uses `agg_L1` (wrong — aggregation prior on oxidation protein)
- `--prior_level WRONG` on `mAb_aggregation` → uses `oxid_L1` (wrong — oxidation prior on aggregation protein)
- `--prior_level blank` → uses `blank` (no mechanism knowledge)

### Per-objective trust diagnostic

The trust diagnostic (`per_objective_trust`) computes a trust score in [0, 1] for each objective independently, using three components:

#### Component 1: LOO calibration (loo_score)

Leave-one-out cross-validation: for each observation, fit a GP on all other observations and predict the held-out point. Compute the z-score (prediction error / prediction uncertainty). If the mean absolute z-score is in [0.3, 1.8], the GP is well-calibrated (loo_score = 1.0). If z-scores are too small (overconfident) or too large (underconfident), the score decreases.

#### Component 2: Lengthscale-to-nearest-neighbour ratio (ls_score)

The GP's fitted lengthscale divided by the mean nearest-neighbour distance between observations. If ls/nn < 0.8, the GP thinks the landscape is rougher than the data spacing — it's extrapolating between points (ls_score = 0.0). If ls/nn > 8.0, the GP thinks the landscape is very smooth — possibly over-smoothing (ls_score = 0.5). In between, the GP is trustworthy (ls_score = 1.0).

**Sparse override**: When `obs_per_dim < 2.0` (i.e., n < 2×d), ls_score is forced to 0.5 (neutral) because the lengthscale estimate is unreliable at this sparsity.

#### Component 3: Uncertainty reduction (unc_score)

Measures how much the GP's predictive uncertainty has been reduced compared to the prior. If uncertainty reduction > 0.3 and LOO z-scores are large (> 1.8), the GP is overconfident (unc_score = 0.0). If uncertainty reduction < 0.1, the GP hasn't learned much (unc_score = 0.5). Otherwise, the GP is genuinely reducing uncertainty (unc_score = 1.0).

#### The disabled noise_score component

An earlier version included a fourth component using `WhiteKernel` noise estimation from a separate GP fit. The idea was to detect per-objective measurement noise (Tm has low noise, kD has high noise) directly from the GP's fitted noise hyperparameter.

**This was verified to NOT work.** The WhiteKernel noise term measures **landscape fitting difficulty**, not measurement noise:

| Objective | True noise_frac | WhiteKernel fitted noise | What it actually measures |
|-----------|----------------|-------------------------|--------------------------|
| Tm | 0.013 (LOW) | 0.344 (HIGH) | Tm's two-weak-bump surface is hard to fit → WhiteKernel absorbs reconstruction error as "noise" |
| kD | 0.096 (HIGH) | 0.000001 (ZERO) | kD's single-mechanism surface fits cleanly → WhiteKernel attributes zero noise |
| visc | 0.100 (HIGH) | 0.394 (HIGH) | Viscosity's nonlinear load surface is hard to fit → high "noise" |

The WhiteKernel **inverts** the true noise ordering: it assigns the most noise to Tm (the lowest-noise objective) and zero noise to kD (the highest-noise objective). This is because kD's surface is dominated by a single strong mechanism (aggregation × anti-agg amino acid) that the Matern kernel fits cleanly, while Tm's surface is a sum of two weak Gaussian bumps that the Matern cannot reconstruct from sparse 16D data.

**Root cause**: The WhiteKernel term is a redundant, confounded version of what ls_score already measures (landscape roughness), with the wrong target for C2. Genuine per-objective measurement noise detection requires replicate measurements (which the oracle doesn't simulate) or an external noise estimate (as Waibel's real data provides via 95% CIs from repeated diffusion measurements).

**Fix**: noise_score = 0.5 (neutral placeholder), WhiteKernel code retained as documentation only.

#### Trust formula

```python
if obs_per_dim < 2.0:  # sparse regime
    trust = 0.6 * loo_score + 0.15 * ls_score + 0.25 * unc_score
else:  # dense regime
    trust = 0.5 * loo_score + 0.3 * ls_score + 0.2 * unc_score
```

At budget=40, d=16: obs_per_dim maxes at 2.5, so the sparse regime applies for 5 of 6 batches. The trust score is 0.700 for all objectives (loo_score=1.0, ls_score=0.5, unc_score=0.5) — the diagnostic cannot differentiate objectives at this sample density. This is an honest limitation, not a bug.

### Mixing weight mechanism

The mixing weight determines how much the LLM vs GP contributes to the final candidate selection:

```python
def mixing_weight(trust_scores, sparse=False):
    w = min(trust_scores.values())
    return (1.0 - w) if sparse else w
```

**Sparse regime inversion**: When `sparse=True` (obs_per_dim < 2.0), the mapping is inverted. A high trust value (0.8) maps to weight=0.2, meaning 80% of selection probability goes to LLM candidates. This is because at low n, the trust score is largely an artifact of loo_score's mean(|z|) statistic being fragile at small n — a single catastrophic LOO miss can be diluted by moderate residuals to a mean|z| that still looks "calibrated." Treating that as "lean on GP" is backwards: the GP is independently confirmed to underperform random at n<30 in d=16. So in the sparse regime, high trust should mean "lean on the LLM," not the GP.

**Dense regime**: When `sparse=False` (obs_per_dim ≥ 2.0), the mapping is direct. High trust → high weight → more GP selection. This is the intuitive direction: when the GP is genuinely trustworthy, use it more.

**Trust trajectory observed in campaigns**: [0.2, 0.2, 0.2, 0.2, 0.2, 0.6] for 6 batches. The first 5 batches are in the sparse regime (weight=0.2 → 80% LLM), and the last batch enters the dense regime (weight=0.6 → 60% GP).

### Diversity filter

The LLM proposals are filtered for diversity in two ways:

1. **Within-batch**: Reject proposals within distance 0.1 of each other in the 16D feature space. This addresses the problem that the LLM's perturb-around-Pareto-front logic produces identical categorical choices (same amino acid/sugar/surfactant) with only small concentration differences.

2. **Cross-batch**: Reject proposals within distance 0.05 of any prior observation (X_obs). This addresses the problem that the LLM re-proposes formulations already tested in earlier batches — verified directly: 2-3 of 5 newly "proposed" formulations were exact duplicates (distance=0.000) of formulations already tested.

The diversity filter does NOT address the softer case where the LLM proposes formulations that are genuinely new input points but converge to similar objective-space outcomes — that is a legitimate exploit-vs-explore tradeoff, not waste.

### Pareto bookkeeping and HV tracking

- **Pareto front**: Computed using vectorised non-dominated sort. All objectives are internally converted to "all maximise" convention (minimisation objectives are negated).
- **Hypervolume**: Computed using pymoo's HV indicator with a **fixed reference point** computed once from the full oracle pool. This avoids the reference-point-drift issue where recomputing the reference every batch makes HV values incomparable across batches.
- **Duplicate filtering**: Candidates within distance 0.05 of existing observations are filtered out before selection.

### Campaign loop

```python
for each batch:
    1. Call strategy_fn(oracle, X_obs, Y_obs, bounds, batch_size, rng, ...)
    2. Snap proposed candidates to nearest pool point
    3. Query the oracle for (Tm, kD, viscosity) at each snapped point
    4. Add new observations to X_obs, Y_obs
    5. Compute Pareto front and hypervolume
    6. Log decision (step, n_obs, pareto_size, hypervolume, trust, mixing_weight)
```

### Cross-seed contamination bug (found and fixed)

The discrete oracle was built once and reused across all 15 seeds. Its `_queried` set (tracking which pool points have been tested) was mutable state on that shared instance. Since `query_mo()` never reset `_queried`, every seed after the first inherited the full set of points queried by all prior seeds — the pool of selectable points shrank monotonically across the 15-seed run.

Additionally, the n_init points were never added to `_queried`, so a seed's first proposal could snap back onto one of its own initial points.

**Fix**: Reset `_queried` at the start of every seed's campaign and register n_init points as already queried.

---

## 10. Phase 6: Bug Discovery and Fixing

Six bugs were identified through systematic verification. They are listed in order of severity and discovery.

### BUG 1: kD not clipped in pool construction (MEDIUM)

**Symptom**: 139/500 pool points had kD < -25.0 (the stated KD_RANGE lower bound), with values extending to -49.5. Tm and viscosity were properly clipped.

**Cause**: The `build()` method clipped Tm to TM_RANGE and viscosity to VISC_RANGE[1]*1.5, but kD had no clip at all. Noise pushed values below the stated range.

**Fix**: Added `kd = np.clip(kd, *KD_RANGE)` in the pool construction.

**Impact**: Low — outlier kD values are in the "bad" direction and won't corrupt the Pareto front. But they produce unrealistic negative kD values that don't match the Waibel anchor.

### BUG 2: Ceiling normalisation erases protein profile effect (CRITICAL)

**Symptom**: kD values were identical across all protein profiles. Arginine was always the best amino acid for kD, regardless of whether the protein was aggregation-prone or oxidation-prone. The wrong-prior stress test was meaningless.

**Cause**: The per-profile ceiling (`kd_ceiling = 0.85*agg + synergy*1.5*agg`) scaled with `aggregation_tendency`, same as the numerator. This algebraic cancellation made `kd_frac` identical across all profiles:

```
kd_frac = kd_raw / kd_ceiling
        = (0.85 * agg * peak) / (0.85 * agg + 1.2 * agg)
        = (0.85 * agg * peak) / (2.05 * agg)
        = 0.415 * peak  ← agg cancels!
```

**Fix**: Use a fixed ceiling of 1.0 (not per-profile). Low-aggregation-tendency profiles then have compressed kD response, which is physically correct — if aggregation isn't this protein's problem, no amount of arginine should make a dramatic difference to kD.

The same class of bug affected Tm. The Tm ceiling was initially per-profile, causing partial cancellation. It was fixed to `TM_CEILING_MAX_ACHIEVABLE = 0.4925` (the maximum achievable `tm_raw` across all profiles, which is enzyme_labile's 0.20×0.40 + 0.55×0.75 = 0.4925).

**Impact**: Critical — without this fix, the wrong-prior stress test cannot work because all profiles produce identical outputs.

### BUG 3: Hypervolume computation incorrect (LOW)

**Symptom**: The oracle's built-in HV used `max(individual_volumes)` instead of the union of dominated hyperrectangles, underestimating true HV by ~28%.

**Fix**: Use pymoo's HV indicator (which computes the proper union). The campaign runner already uses pymoo HV, so this only affected the oracle's internal analysis tools.

### BUG 4: Trust diagnostic collapses to identical values (CRITICAL)

**Symptom**: At campaign-relevant budgets (n=10, d=16), the trust diagnostic returned 0.800 for ALL three objectives, making it impossible to differentiate which objectives the GP models well vs poorly.

**Cause**: With n=10, d=16 → obs_per_dim = 0.625 < 2.0, triggering the sparse override:
- loo_score = 1.0 for all (LOO |z| in [0.3, 1.8] — all well-calibrated)
- ls_score = 0.5 for all (sparse override forces 0.5)
- unc_score = 0.5 for all (uncertainty_reduction < 0.1)
- trust = 0.6×1.0 + 0.15×0.5 + 0.25×0.5 = 0.700 for all

**Deeper issue**: The diagnostic measures landscape roughness (ls/nn ratio), not measurement noise (noise_frac). C2 requires detecting that kD has high measurement noise (0.096) while Tm has low (0.012). The GP's internal noise term absorbs measurement noise, so LOO calibration looks similar regardless.

**Attempted fix (WhiteKernel)**: Added a WhiteKernel noise term to detect per-objective measurement noise. This was verified to NOT work — the WhiteKernel inverts the true noise ordering (see "The disabled noise_score component" above).

**Final fix**: Remove the noise_score component entirely. Revert to the honest 3-component diagnostic. Document that per-objective measurement noise differentiation is not achievable from GP-internal quantities alone at this sample density.

**Impact**: Critical for C2 — the trust diagnostic cannot differentiate objectives at budget=40/d=16. This is an honest limitation, not a fixable bug. To validate C2, either increase the budget (≥80 to reach obs_per_dim ≥ 3.0) or reduce dimensionality.

### BUG 5: Mixing weight has negligible effect (MEDIUM)

**Symptom**: The mixing weight (`weight = min(trust_scores.values())`) had almost no effect on candidate selection because the final selection from the combined LLM+GP pool was uniform random, not weighted by trust.

**Analysis**: Even with weight=0.3 (low trust in GP) vs weight=0.8 (high trust), the pool composition differed by only ~1 candidate slot, and uniform random selection then discarded even that weak difference.

**Fix**: Replace uniform random selection with trust-weighted selection probability:
```python
gp_prob_each = weight / n_gp_in_pool
llm_prob_each = (1.0 - weight) / n_llm_in_pool
probs = np.where(source_is_llm, llm_prob_each, gp_prob_each)
selected_idx = rng.choice(len(pool), n_select, replace=False, p=probs)
```

Now weight=0.8 means 80% of selection probability goes to GP candidates, weight=0.2 means 80% goes to LLM candidates.

### BUG 6: HV reference point drifts between batches (LOW)

**Symptom**: The HV reference point was recomputed each batch from current observations. As better points were found, the reference shifted upward (e.g., the kD component shifted by 26.5 units = 42% change), making batch-to-batch HV values incomparable.

**Fix**: Compute the reference point once at campaign start from the full oracle pool and use it for all batches.

---

## 11. Phase 7: Multi-Objective Campaign Results

### mAb_aggregation, L1 prior, budget=40, 15 seeds, real LLM (qwen2.5:72b-instruct)

| Condition | HV (mean ± SD) | Pareto size | HV/point |
|-----------|---------------|-------------|----------|
| mo_egbo_real | 13699.7 ± 1003.3 | 10.7 | 1284.3 |
| mo_llm | 13401.1 ± 988.7 | 8.5 | 1582.8 |
| mo_random | 11472.3 ± 1556.4 | 8.1 | 1410.5 |
| mo_egbo (lightweight) | 10181.1 ± 2605.8 | 10.7 | 948.6 |

### Interpretation

1. **mo_egbo_real and mo_llm both clearly beat mo_random** (Δ ~2200, ~2 SD). This validates C1 in multi-objective form: informed strategies beat random on mAb_aggregation.

2. **mo_egbo_real and mo_llm are nearly tied** (Δ=299, within 1 SD). The BoTorch qLogNEHVI acquisition and the LLM with correct prior are equally effective on this profile.

3. **mo_llm has the highest HV per point** (1582.8) — it finds fewer Pareto points but each one is more valuable (better spread in objective space). This is the diversity filter working: it prevents the LLM from clustering proposals.

4. **Lightweight mo_egbo significantly underperforms** (10181 vs 13699 for real EGBO). The UCB-based Pareto ranking is noise-dominated at low n. This confirms the need for proper multi-objective acquisition functions (qLogNEHVI).

5. **Trust trajectory**: [0.2, 0.2, 0.2, 0.2, 0.2, 0.6] — sparse inversion working correctly. The first 5 batches give 80% weight to LLM, the last batch gives 60% to GP.

6. **LLM proposal target distribution**: Tm=44%, kD=34%, viscosity=22%. Slightly Tm-skewed, which is reasonable for an aggregation-prone protein where Tm and kD are both important.

### What this validated

- **C1 (multi-objective)**: Informed strategies (egbo_real, llm) beat random on mAb_aggregation. Validated.
- **C3 (mixing weight)**: The sparse-inverted mixing weight produces sensible behaviour (80% LLM in sparse regime). The mechanism works as designed, though it cannot be fully validated at this budget because trust never differentiates across objectives.

### What was NOT validated

- **C2**: Trust diagnostic doesn't differentiate at budget=40/d=16 (all objectives get trust=0.700). Needs budget=80.
- **C1 robustness**: Only tested on mAb_aggregation. Need other profiles.
- **C3 full validation**: The mixing weight is constant (0.2) for 5/6 batches. Need to see it actually adapt.

---

## 12. Phase 8: Wrong-Prior Stress Test in Multi-Objective

### What was tested

The wrong-prior stress test was extended to the multi-objective oracle. The key question: does the clean single-objective result (WRONG < random < EGBO < L1) replicate when objectives are separate rather than collapsed?

### mAb_oxidation results (budget=40, 15 seeds, real LLM)

| Prior | mo_llm | mo_egbo_real | mo_random | Ordering |
|-------|--------|-------------|-----------|----------|
| WRONG (agg_L1) | **2763.3 ± 234.9** | — | — | WRONG >> everything |
| L1 (oxid_L1, correct) | 2413.1 ± 338.0 | 2527.4 ± 385.0 | 2301.4 ± 308.7 | egbo ≈ llm ≈ random |
| blank | 2422.1 ± 367.2 | 2443.6 ± 395.1 | 2301.4 ± 308.7 | egbo ≈ llm ≈ random |

### The complete ranking

```
WRONG (2763) > egbo_real (2444) ≈ blank_llm (2422) ≈ L1_llm (2413) > random (2301)
```

### Finding 1: No informed strategy beats random on mAb_oxidation

mo_egbo_real (2527) vs mo_random (2301): Δ=226, but with SDs of 385 and 309, this is not significant. mo_llm with the correct prior (2413) is indistinguishable from blank (2422) and from random. The entire oxidation landscape is hard at budget=40/d=16 — not just for the LLM, for every method.

This is consistent with the single-objective result: mAb_oxidation was the weakest case there too (EGBO already reached AUC=0.900, leaving little room for improvement). The multi-objective version amplifies the problem because Tm only fills 38.1% of its range (max 65.7°C) — the objective that should differentiate correct from wrong priors has almost no dynamic range on this profile.

### Finding 2: The WRONG prior wins — and the mechanism is verified

The WRONG prior (agg_L1) tells the LLM to use arginine. On mAb_oxidation:

- **Arginine gives kD = +13.4 mL/g** (51.2% of kD range, because kD's scoring formula rewards anti-aggregation amino acids with `0.85 * aggregation_tendency * aa_peak`, and even at agg=0.25, this produces a meaningful kD improvement)
- **Methionine gives kD = -24.1 mL/g** (1.2% of kD range, because methionine is not anti-aggregation, so its kD contribution is only `0.05 * aggregation_tendency * aa_peak`)
- **kD advantage of WRONG over CORRECT: 37.5 mL/g** — enormous

Meanwhile the Tm penalty of using arginine instead of methionine is only ~0.5°C (because Tm is dominated by sugar contribution, and both priors recommend sugars). The WRONG prior wins because it's accidentally correct about the objective (kD) that has the most room to improve.

This was verified directly from the oracle code. The kD scoring formula is:
```python
kd_aa_contrib = np.where(
    aa_is_antiagg, 0.85 * aggregation_tendency * kd_aa_peak,  # arginine
    0.05 * aggregation_tendency * kd_aa_peak)                 # methionine
```

At agg=0.25, arginine's kD contribution is 0.85×0.25 = 0.2125, which translates to kD = -25 + 0.5125×75 = +13.4. The ceiling is fixed at 1.0, so this 51.2% fill is a genuine, physically-motivated compression.

### Finding 3: Single-objective vs multi-objective tell different stories

| | Single-objective (collapsed score) | Multi-objective (3 separate objectives) |
|---|---|---|
| mAb_oxidation WRONG | AUC=0.817, **worse than random** (0.860) | HV=2763, **better than everything** |
| Mechanism | Collapsed score dominated by oxidation pathway → arginine hurts overall | kD is a separate objective → arginine helps kD enormously, HV rewards it |

The single-objective oracle collapsed Tm/kD/viscosity into one score weighted by the correct pathway. This made "wrong pathway" advice uniformly harmful because the collapsed score couldn't reward arginine's kD improvement independently of its irrelevance to oxidation. The multi-objective oracle separates the objectives, and hypervolume rewards any genuine improvement on any axis — so arginine's kD benefit shows up even when the mechanism story is wrong.

### Is this an oracle bug?

**No.** The oracle is correct that arginine improves kD on mAb_oxidation. In reality, arginine does improve kD on virtually every mAb, including oxidation-prone ones — the kD interaction parameter measures colloidal stability, which is mechanistically linked to protein-protein attraction regardless of which degradation pathway dominates.

What changed between single-objective and multi-objective is not the oracle — it's the **reward structure**. The single-objective collapsed score implicitly weighted objectives by pathway relevance, making wrong-pathway advice uniformly harmful. The multi-objective HV rewards any improvement on any axis, making wrong-pathway advice partially helpful when it's correct about a specific excipient-objective relationship.

This is a genuine property of multi-objective optimisation, not an artefact. In a real formulation lab, if you tell a scientist "this protein aggregates, use arginine" and the protein actually oxidises, the arginine still improves kD. The advice is wrong about the mechanism narrative but partially correct about a specific excipient effect.

### What this means for the claims

**C3 reframing**: The original formulation — "wrong mechanism knowledge hurts performance" — holds in single-objective but inverts in multi-objective. The correct reframing:

> "In multi-objective formulation, wrong-prior harm is attenuated because shared excipient effects create partial correctness on individual objectives. Hypervolume rewards this partial correctness, so a prior that's wrong about the mechanism narrative but correct about one excipient-objective relationship can outperform a correct-but-vague prior."

This is a stronger and more nuanced claim than the original C3. It says the concept of "wrong prior" is less binary in multi-objective settings — priors are wrong about *narrative* but can be partially correct about *specific relationships*, and the multi-objective reward structure reveals this.

### The critical missing data point

mAb_aggregation has NOT been tested with WRONG or blank priors in multi-objective form. On mAb_aggregation, the L1 prior is arginine-forward (correct for aggregation). If the WRONG prior (oxidation-focused, methionine-forward) also performs well on mAb_aggregation because arginine still dominates kD there, then the mAb_aggregation result was never about mechanism-matching — it was about arginine being the dominant kD lever on every profile.

This is the single most informative comparison remaining.

---

## 13. The Four Claims — Status Assessment

| Claim | Status | Evidence | What's needed |
|-------|--------|----------|---------------|
| **C1**: LLM knowledge beats GP-only | **Validated (SO), partial (MO)** | SO: clean gradient on mAb_agg (L1 > EGBO > random > WRONG). MO: egbo_real ≈ llm >> random on mAb_aggregation. But MO null on mAb_oxidation (no strategy beats random). | mAb_aggregation with WRONG/blank to confirm the MO result is about mechanism, not arginine's cross-profile kD dominance |
| **C2**: Per-objective trust diagnostic | **Not validated** | Trust collapses to 0.700 for all objectives at budget=40/d=16. WhiteKernel noise detection inverts true noise ordering. | Budget=80 to reach obs_per_dim ≥ 3.0 where ls_score differentiates. Or replicate simulation for genuine noise estimation. |
| **C3**: Mixing weight mechanism | **Partially validated, reframed** | Sparse inversion works (weight=0.2 → 80% LLM). But weight is constant for 5/6 batches. Wrong-prior result reframes C3: wrong priors can partially help in MO. | Budget=80 to see mixing weight actually adapt. mAb_aggregation WRONG/blank to test the reframed C3. |
| **C4**: End-to-end pipeline | **Partially validated** | Campaign runs end-to-end with real LLM. Pareto bookkeeping, diversity filter, trust-weighted selection all work. | Test on real Waibel data (33 formulations). Test on other profiles. |

---

## 14. Literature Context

### Directly related papers

1. **Aqeeli, Leelawat, Shorthouse (2026)** — "Novelty-aware evolutionary Bayesian optimisation for multi-objective discovery science" [1]. From the same lab (UCL, David Shorthouse). Combines evolutionary algorithms with BO and introduces novelty-aware selection to address MOBO's limited exploration and reduced diversity. This is the same Pareto diversity problem we encountered (mo_llm's inflated-but-weak front) — they fix it at the acquisition function level, we fix it at the strategy selection level.

2. **Waibel et al. (2025)** — "Bayesian Optimization for Efficient Multiobjective Formulation Development of Biologics" [3]. The source of our oracle dataset. Optimised bococizumab-IgG1 formulation with 3 objectives (Tm, kD, RM_Agi) in 33 experiments. Used sequential MOBO with exploitation (Pareto front from GP models) and exploration (distance-based) strategies, with greedy Kriging Believer for parallel execution. Key finding: BO found highly optimised formulations in 33 experiments — several orders of magnitude fewer than full screen, half as many as simplest DOE.

3. **Liu et al. (2026)** — "Fast-tracking complex formulation development with multi-objective Bayesian optimisation" [4]. Also from the Shorthouse group. Thermosensitive in-situ nasal gel for sertraline, 4 excipients, 3 objectives (maximise % release, maintain gelation temperature, minimise micelle size). 14-22 MOBO experiments vs 81 full factorial DoE. Post-hoc model diagnostics revealed excipient lengthscale, predictive accuracy, and noise management patterns — but these are post-hoc, used for interpretation after the campaign. Our project's contribution is making these diagnostics proactive.

4. **Ros, Chan, Cook, Shorthouse (2026)** — "Artificial intelligence and machine learning guided optimization in drug delivery" [5]. Review covering surrogate modelling, BO, active learning, and MOBO for drug delivery. Discusses the exploration-exploitation tradeoff, scalarisation vs Pareto-based acquisition functions, and the emergence of self-driving laboratories.

### How our work relates

| What existing papers do | What our project does differently |
|---|---|
| Single MOBO strategy for entire campaign | Adaptive strategy selection based on trust diagnostics |
| Post-hoc model diagnostics | Real-time trust scoring that drives mixing weights |
| No LLM/prior knowledge integration | LLM proposals as alternative when GP is untrustworthy |
| Fixed exploration-exploitation balance | Sparse-inverted mixing weight that shifts GP↔LLM ratio |
| One formulation type per paper | Cross-profile generalisation (4 protein profiles) |
| Diversity fixed at acquisition level (Aqeeli) | Diversity fixed at strategy selection level (our approach) |

---

## 15. Future Directions: RL Meta-Controller

### The idea

Instead of using a heuristic trust diagnostic → mixing weight pipeline, train a reinforcement learning (RL) agent to learn the optimal strategy selection policy directly from simulated campaigns.

### Why it fits

The existing oracle + campaign simulator **is** the RL environment — a Markov decision process where:
- **State**: campaign trajectory (observations, trust scores, budget remaining, current Pareto front, obs_per_dim)
- **Action**: select strategy for next batch (LLM, EGBO, random, exploit, explore)
- **Reward**: hypervolume improvement per batch
- **Environment**: the oracle + campaign simulator, run for thousands of episodes

This is **meta-RL**: the agent isn't learning to propose formulations (BO already does that), it's learning to *select the strategy that proposes formulations*.

### Why it's NOT "RL replaces BO"

BO acquisition functions (qLogNEHVI, UCB, EI) are already near-optimal policies under GP assumptions. An RL agent would need enormous training to match what qLogNEHVI does analytically. The value isn't in replacing the acquisition function — it's in choosing *which* acquisition function (or non-BO strategy) to deploy.

### Feasible steps

1. **Formalise the MDP** (1-2 days): Define state-action-reward tuple. Key design choice: episode length. At budget=40 with batch_size=5, you get 8 decision points — very short. May need budget=80-120 for RL training.

2. **Build the RL training loop** (3-5 days): Wrap the campaign simulator in a Gymnasium interface. Train with PPO or SAC (stable-baselines3). Run 10,000-50,000 episodes — cheap because the oracle is fast.

3. **Curriculum across profiles** (1-2 days): Train on easy profiles first (enzyme_labile — full Tm range), then hard ones (mAb_oxidation — narrow Tm range).

4. **Compare RL meta-controller vs heuristic trust** (2-3 days): Run held-out campaigns with fixed strategies, heuristic adaptive, and RL adaptive. If the RL agent discovers the same sparse-inversion rule, that validates the heuristic. If it discovers something different, that's a new design rule.

5. **Transfer/generalisation test** (2-3 days): Train on one profile, test on another. If the policy transfers, the strategy rules are profile-independent (strong claim for C4).

### Biggest risk

**Overfitting**: the RL agent memorises the oracle's specific landscape rather than learning generalisable strategy rules. Domain randomisation across profiles, noise levels, and ceiling values is essential.

### Recommended sequence

1. Finish the wrong-prior stress test (mAb_aggregation with WRONG/blank)
2. Run budget=80 campaigns (validates C2/C3)
3. Then build the RL meta-controller on top of the validated simulator

---

## 16. Complete File Inventory

### Core code files

| File | Size | Description |
|------|------|-------------|
| `excipient_oracle.py` | 28,801 bytes | Single-objective excipient oracle (original) |
| `excipient_oracle_mo.py` | 28,802 bytes | Multi-objective oracle (Tm, kD, viscosity) |
| `excipient_campaign_mo.py` | 52,746 bytes | Multi-objective campaign runner (4 strategies, trust, mixing) |
| `analyse_mo_campaign_logs.py` | 10,709 bytes | Campaign log analysis script |
| `posthoc_llm_comparison_v2.py` | 36,971 bytes | LLM-BO comparison framework (v2, with JSON fixes) |
| `excipient_adapter.py` | 912 bytes | Adapter for oracle integration |

### Data files

| File | Description |
|------|-------------|
| `waibel_mAb_formulation_dataset_CORRECTED.csv` | 33 formulations, 7 inputs, 3 objectives (kD sign-corrected) |
| `mp5c00591_si_001.pdf` | Waibel et al. Supporting Information (source) |

### Result files

| File | Description |
|------|-------------|
| `evaluation_excipient_oracle_mo.md` | Oracle evaluation report |
| `evaluation_excipient_campaign_mo.md` | Campaign evaluation report |
| `project_log_post_domain_shift.md` | Earlier project log (pre-MO work) |
| `execution_trace/PLAN.md` | Plan document (bug fixes) |

### Design documents

| File | Description |
|------|-------------|
| `mechanistic_diagnosis.md` | Mechanistic diagnosis of when LLM priors help vs hurt |
| `paper_narrative_detailed.md` | Detailed paper narrative |
| `protein_bo_phase1_design.md` | Phase 1 experimental design |
| `excipient_llm_prompt_template.md` | LLM prompt template |
| `transfer_learning_dataset_triples.md` | Transfer learning design |

### Earlier infrastructure (pre-pivot, still relevant)

| File | Description |
|------|-------------|
| `campaign_orchestrator.py` | Campaign orchestration |
| `design_rules_analysis.py` | Design rule extraction |
| `mid_campaign_analysis.py` | Mid-campaign analysis |
| `feasibility_agent.py` | Feasibility checking |
| `scoring_agent.py` | Scoring agent |
| `sdl_formulation_lab/` | SDL formulation lab scenarios |
| `llm_bo_benchmarks.py` | LLM-BO benchmark suite |

---

## 17. Key Numbers Reference

### Oracle parameters

| Parameter | Value |
|-----------|-------|
| Input dimensions | 16D (5 AA one-hot + conc + 4 sugar one-hot + conc + 3 surf one-hot + conc + EDTA) |
| Discrete pool size | 500 |
| TM_RANGE | (60.0, 75.0) °C |
| KD_RANGE | (-25.0, 50.0) mL/g |
| VISC_RANGE | (2.0, 25.0) cP |
| TM_CEILING_MAX_ACHIEVABLE | 0.4925 |
| kd_ceiling | 1.0 (fixed) |
| Default noise | tm=0.013, kd=0.096, visc=0.10 (Waibel-anchored) |

### Protein profiles

| Profile | agg | oxid | denat | metal |
|---------|-----|------|-------|-------|
| mAb_aggregation | 0.85 | 0.20 | 0.30 | 0.15 |
| mAb_oxidation | 0.25 | 0.80 | 0.25 | 0.60 |
| enzyme_labile | 0.40 | 0.30 | 0.75 | 0.20 |
| mixed | 0.55 | 0.45 | 0.45 | 0.35 |

### Per-profile objective range filling

| Profile | Tm max | Tm fill % | kD max (Arg) | kD fill % | kD max (Met) |
|---------|--------|-----------|--------------|-----------|--------------|
| mAb_aggregation | 70.2°C | 68.0% | +50.0 | 100.0% | -21.8 |
| mAb_oxidation | 65.7°C | 38.1% | +13.4 | 51.2% | -24.1 |
| enzyme_labile | 75.0°C | 100.0% | +36.5 | 82.0% | -23.5 |
| mixed | 70.9°C | 72.6% | +50.0 | 100.0% | -22.9 |

### Single-objective stress test (AUC, 20 seeds)

| Profile | L1 | L4 (blank) | WRONG | EGBO | random |
|---------|-----|-----------|-------|------|--------|
| mAb_aggregation | 0.863 | 0.837 | 0.708 | 0.818 | 0.749 |
| mAb_oxidation | 0.917 | 0.889 | 0.817 | 0.900 | 0.860 |

### Multi-objective campaign results (HV, 15 seeds, budget=40, real LLM)

| Profile | Prior | mo_llm | mo_egbo_real | mo_random |
|---------|-------|--------|-------------|-----------|
| mAb_aggregation | L1 | 13401 ± 989 | 13699 ± 1003 | 11472 ± 1556 |
| mAb_oxidation | L1 | 2413 ± 338 | 2527 ± 385 | 2301 ± 309 |
| mAb_oxidation | blank | 2422 ± 367 | 2444 ± 395 | 2301 ± 309 |
| mAb_oxidation | WRONG | **2763 ± 235** | — | — |

### Trust diagnostic

| Component | What it measures | Sparse override |
|-----------|-----------------|-----------------|
| loo_score | Leave-one-out calibration (z-score) | No (always computed) |
| ls_score | Lengthscale/nearest-neighbour ratio | 0.5 when obs_per_dim < 2.0 |
| unc_score | Uncertainty reduction vs prior | No |
| noise_score | DISABLED (WhiteKernel inverts true noise) | 0.5 (placeholder) |

Trust formula: sparse: 0.6×loo + 0.15×ls + 0.25×unc; dense: 0.5×loo + 0.3×ls + 0.2×unc

### Mixing weight

| Regime | Formula | Example |
|--------|---------|---------|
| Sparse (obs_per_dim < 2.0) | weight = 1 - min(trust) | trust=0.8 → weight=0.2 → 80% LLM |
| Dense (obs_per_dim ≥ 2.0) | weight = min(trust) | trust=0.6 → weight=0.6 → 60% GP |

### Waibel dataset

| Metric | Value |
|--------|-------|
| Formulations | 33 |
| Varying inputs | 7 |
| Objectives | 3 (Tm, kD, RM_Agi) |
| Tm noise-to-signal | 0.012 (HIGH trust) |
| kD noise-to-signal | 0.096 (LOW trust) |
| RM_Agi noise-to-signal | 0.061 (MEDIUM trust) |
| Pareto-optimal points | 5 |
| Campaign structure | 15 DoE + 18 BO = 33 total |

---

## References

[1] Aqeeli, M., Leelawat, T., Shorthouse, D. "Novelty-aware evolutionary Bayesian optimisation for multi-objective discovery science." Digital Discovery, 2026. DOI: 10.1039/d6dd00134c

[2] Ros, H., Abdalla, Y., Cook, M.T., Shorthouse, D. "Efficient discovery of new medicine formulations using a semi-self-driven robotic formulator." Digital Discovery, 2025, 4(8), 2263-2272. DOI: 10.1039/d5dd00171d

[3] Waibel, I., Schneider, T.N., Fischer, F.J., et al. "Bayesian Optimization for Efficient Multiobjective Formulation Development of Biologics." Molecular Pharmaceutics, 2025. DOI: 10.1021/acs.molpharmaceut.5c00591

[4] Liu, H., Gucic, A., Liu, H., Cook, M.T., Shorthouse, D. "Fast-tracking complex formulation development with multi-objective Bayesian optimisation." Journal of Controlled Release, 2026. DOI: 10.1016/j.jconrel.2026.115171

[5] Ros, H., Chan, N., Cook, M.T., Shorthouse, D. "Artificial intelligence and machine learning guided optimization in drug delivery." Advanced Drug Delivery Reviews, 2026. DOI: 10.1016/j.addr.2026.115781
