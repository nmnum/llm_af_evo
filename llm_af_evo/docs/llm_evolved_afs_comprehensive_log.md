# LLM-Evolved Acquisition Functions: Comprehensive Project Log

**Project:** Self-Driving Lab — Adaptive Campaign Planner
**Author:** Neha Mungale
**Date:** 2026-08-01
**Scope:** Everything done on LLM-evolved acquisition functions (AFs), including successful runs, failed runs, dead ends, bug analyses, and design rules derived.

**Verification note (added 2026-08-01):** This document was produced in a separate
session (evidenced by its `/mnt/results/`, "user uploads", and `worker-0.ipynb`
references) and then cross-checked here directly against the files on disk in
`evolution_runs/` and `af_code_logs/`. Spot-checked and **confirmed exact**: the
Run 1/2/4 fitness tables (§5, §6, §8) against `final_population.json`; the
`noisy_front_hvi` repetition-loop bug (§9.2, 54/71 comment lines, no return) against
`call_00004.py`; the `local_penalization` triple-bug (§9.3) against `call_00007.py`;
the `pareto_membership` inverted-dominance bug (§9.4) against `call_00008.py`; and
the `mab_noise_diagnostic_results.json` structure (§12). No fabricated numbers were
found in any of these checks.

**One nuance added by verification, not in the original doc:** §9.2 attributes the
`noisy_front_hvi` repetition-loop bug to Run 4 (`run_v2_coatings_gamma001_v2`,
Jul 31 07:07). The `af_interface_v2.py` source comment for this hint says it was
"rewritten (v2.2) after a first attempt... produced a broken non-dominated filter
and an absolute-HV-not-improvement bug" — but that comment refers to a **different,
earlier, 9-call smoke run**, `run_v2_coatings_gamma001` (no `_v2` suffix, Jul 30
22:31), whose own `call_00004.py` has a *different* bug (calls
`hypercube_volume(candidate_objectives, ref_point)` directly — the exact
absolute-volume mistake the rewritten hint text warns against — and does actually
`return` a value). So the full picture is: the hint was rewritten once after the
first bug, and the LLM's very next real attempt (in the `_v2` run analyzed in this
document) failed in a **new, different way** (comment-spiral, no return at all)
rather than repeating the original bug. Two distinct `noisy_front_hvi` failure
modes exist across the two runs, not one bug fixed then cleanly reproduced.

For the earlier (pre-v2, through Jul 27) fitness-generation history (2a→2b→margin),
the 8 mechanism gates, and the compose_batch/DA-COREG dead ends that this document's
§15 only briefly summarizes, see the companion document [SESSION_LOG.md](SESSION_LOG.md)
in this same directory.

---

## Table of Contents

1. [What This Is and Why](#1-what-this-is-and-why)
2. [The AF Evolution Framework](#2-the-af-evolution-framework)
3. [The Two Oracle Domains](#3-the-two-oracle-domains)
4. [The Hint Set Evolution](#4-the-hint-set-evolution)
5. [Run 1: Coatings, gamma=0.005, n=15](#5-run-1-coatings-gamma0005-n15)
6. [Run 2: Coatings, gamma=0.005, n=75](#6-run-2-coatings-gamma0005-n75)
7. [Run 3: mAb, gamma=0.005, n=75](#7-run-3-mab-gamma0005-n75)
8. [Run 4: Coatings, gamma=0.001, n=75, 8-Hint Set](#8-run-4-coatings-gamma0001-n75-8-hint-set)
9. [The Four New Hints: Bug Analysis](#9-the-four-new-hints-bug-analysis)
10. [Held-Out Validation Results](#10-held-out-validation-results)
11. [Cross-Domain Transfer Tests](#11-cross-domain-transfer-tests)
12. [The Noise Diagnostic](#12-the-noise-diagnostic)
13. [The 3-Seed mAb Re-Validation](#13-the-3-seed-mab-re-validation)
14. [Synthetic Benchmark Generalization](#14-synthetic-benchmark-generalization)
15. [Dead Ends and Negative Results (Pre-Evolution Work)](#15-dead-ends-and-negative-results-pre-evolution-work)
16. [Design Rules Derived](#16-design-rules-derived)
17. [First-Principles Redesign for Real-World Data](#17-first-principles-redesign-for-real-world-data)
18. [Real mAb Formulation Data Landscape](#18-real-mab-formulation-data-landscape)
19. [User Corrections Log](#19-user-corrections-log)
20. [Current Plan and Next Steps](#20-current-plan-and-next-steps)

---

## 1. What This Is and Why

### The Problem

Self-driving laboratories (SDLs) run autonomous experimental campaigns: a Bayesian optimisation (BO) loop repeatedly fits a surrogate model (typically a Gaussian Process, GP) to observed data, uses an acquisition function (AF) to rank candidate experiments, picks a batch, runs them, and repeats. The AF encodes the exploration-exploitation trade-off — it decides what to try next.

Most SDL campaigns use a fixed AF throughout (e.g. qEHVI, qNEHVI, UCB). But the optimal trade-off changes over a campaign's lifetime: early on, you want broad exploration; later, you want exploitation near the Pareto front. The question this project asks is: **can an LLM evolve better acquisition functions than the fixed ones we use by default, and can we derive design rules for when different AF strategies work?**

### What "LLM-Evolved AFs" Means

Rather than hand-designing a new AF, we use an evolutionary loop where:

1. A population of AF programs (Python functions called `score_pool`) is initialised from a set of "seed" programs and "strategy hints" (one-line descriptions of mechanisms the LLM should implement).
2. Each AF is evaluated by running it through simulated experimental campaigns against a baseline (EGBO with novelty selection). The AF that produces higher hypervolume (HV) margins wins.
3. The LLM is asked to mutate/rewrite the best-performing AFs to produce the next generation.
4. This repeats for a fixed number of generations (20 in all our runs).

This is inspired by FunBO/FunSearch/AlphaEvolve-style program synthesis, where an LLM generates code and an evolutionary loop selects for performance. The key difference from those systems: our fitness function is not a unit test but a full multi-objective BO campaign simulation.

### What This Document Covers

This is a complete log of every evolution run, every validation test, every dead end, every bug found, and every design rule extracted. It includes runs that stagnated immediately, runs that progressed, hints that produced broken code, and the noise analysis that explained why the mAb evolution couldn't work. The goal is to have a single reference document that captures the full state of the LLM-evolved AF work.

---

## 2. The AF Evolution Framework

### The AF Interface Contract

Every AF is a Python function with this exact signature:

```python
def score_pool(context):
    # ... compute scores ...
    return scores  # list of floats, one per candidate in the pool
```

The `context` dictionary contains:

| Key | Type | Description |
|-----|------|-------------|
| `pool` | list of dicts | Candidate experiments, each with `x` (input vector) and `gp_posterior` |
| `pool[i].x` | np.array | The candidate's input vector (D-dimensional) |
| `pool[i].gp_posterior` | dict | Per-objective GP predictions: `{obj_name: {"mean": float, "std": float}}` |
| `objective_names` | list of str | Names of the objectives (e.g. `["conductivity", "conductance_std"]`) |
| `objective_directions` | list of str | May or may not be present. `"max"` or `"min"` per objective |
| `pareto_front_range` | float or dict | Range of the current Pareto front per objective. Used for normalising sigma. If dict, indexed by objective name |
| `pareto_front` | list | Current non-dominated set (list of objective-space points) |
| `X_obs` | np.array | Observed inputs so far (N × D) |
| `ref_point` | np.array | Hypervolume reference point |
| `campaign` | dict | Campaign state: `step`, `budget`, `progress` (0→1), `stagnant_batches` |

**Critical convention:** `gp[name]["mean"]` is ALREADY in all-maximise convention by the time `score_pool` sees it. The upstream strategy code flips min-objectives before building the context. Re-applying `objective_directions` inside `score_pool` would double-flip and invert behaviour on any min-objective. This is documented in the `trust_only` seed's docstring as a warning.

**What was NOT available in the context initially:** `Y_obs` (observed outputs). This was needed by the `noisy_front_hvi` and `outcome_novelty` hints (v2.1). Adding it is a one-line change in the strategy code that builds context — it does not alter what is evolved (score_pool still only ranks a fixed pool).

### The Evolutionary Loop

The evolution uses a (µ, λ) strategy:

- **Population size (µ):** 8
- **Offspring per generation (λ):** 8
- **Generations:** 20 (fixed)
- **LLM calls per generation:** 2 (each produces 4 offspring via mutation/recombination)
- **Total LLM calls per run:** 40
- **LLM failures across all runs:** 0 (the LLM always produced syntactically valid Python)

At each generation:
1. The top µ AFs by fitness are selected as parents.
2. The LLM is prompted with 2 parent AFs and asked to produce mutated/recombined offspring.
3. All λ offspring + µ parents are evaluated.
4. The top µ by fitness survive to the next generation.

### The Fitness Function

Fitness measures how much better an AF is than the baseline (EGBO with novelty selection, i.e. `mo_egbo_novelty`):

```python
# For each campaign, compute the relative HV margin:
rel_margin = (hv_af - hv_baseline) / abs(hv_baseline)

# Fitness = mean margin - gamma * LOC
fitness = mean(rel_margins) - gamma * loc
```

Where:
- `hv_af` is the final hypervolume achieved by the AF in that campaign
- `hv_baseline` is the final hypervolume achieved by `mo_egbo_novelty` in the same campaign
- `rel_margins` is the list of relative margins across all training campaigns
- `gamma` is the LOC (lines of code) penalty coefficient
- `loc` is the number of lines in the AF's `score_pool` function

The margin is **relative** (a fraction), not raw HV. This was a correction the user made early on — the initial analysis incorrectly treated it as raw HV, which led to wrong conclusions about whether gamma=0.005 was "inert."

### The LOC Penalty (gamma)

The LOC penalty serves as a parsimony pressure: simpler AFs are preferred over complex ones, all else being equal. The coefficient `gamma` controls how aggressively:

- **gamma = 0.005** (Runs 1, 2, 3): A 10-line AF pays a penalty of 0.05. On coatings, where margins are ~3-5%, this penalty exceeds the margin difference between most AFs. Result: evolution stagnates immediately because the LOC penalty overrules higher-margin but higher-LOC AFs.
- **gamma = 0.001** (Run 4): A 10-line AF pays a penalty of 0.01. This is small enough that a 0.37% margin improvement (gen6_child0 over trust_only) can overcome the 3-line LOC difference. Result: evolution progresses for one generation before stagnating.

The gamma value is domain-dependent: it should be calibrated relative to the typical margin magnitude. On coatings (margins ~3-4%), gamma=0.001 works. On mAb (margins ~4-5% but with CV=50% noise), the fitness is too noisy for gamma to matter — the noise must be fixed first.

### Campaign Parameters

All campaigns use the same short-horizon setup:

| Parameter | Value |
|-----------|-------|
| Budget | 40 experiments |
| Initial design | 10 (LHS or random) |
| Batch size | 5 |
| Batches | 6 (after initial design) |
| GP | Per-objective Matern 5/2 (BoTorch SingleTaskGP) |
| Candidate generation | U-NSGA-III (pymoo) |
| Baseline AF | qLogNEHVI + novelty (mo_egbo_novelty) |

This is deliberately short-horizon (6 batches of 5 = 30 sequential decisions after a 10-point initial design). Real SDL campaigns are short — you don't get 200 experiments. This constraint shapes everything: the AF must make good decisions early, and the evolution must find improvements that manifest within 40 experiments.

---

## 3. The Two Oracle Domains

### Coatings Oracle

- **Source:** Real Ada coatings dataset (pareto_20210112), 253 samples
- **Inputs:** 4D continuous (concentration, temperature, flow rate, pressure)
- **Objectives:** `conductivity` (maximise), `conductance_std` (minimise)
- **Pareto front:** 79/253 = 31.2%
- **Objective correlation:** r = 0.0032 (near-independent — the two objectives don't trade off predictably)
- **HV scale:** ~400-450 (reference point dependent)
- **Noise:** Moderate. The GP-fit seed and U-NSGA-III seed contribute ~29% of total variance (see Noise Diagnostic below).

The coatings domain is the "easier" substrate: the GP can model the landscape well, the objectives are near-independent (so the Pareto front is broad), and the AF's uncertainty term (sigma) doesn't change candidate rankings much. This last property — **mu_sum dominance** — is the most important finding from the coatings runs.

### mAb Excipient Oracle

- **Source:** Simulated antibody formulation oracle (excipient catalogue)
- **Inputs:** 16D encoding (5 amino acids, 4 sugars, 3 surfactants, EDTA boolean)
- **Objectives:** Tm (maximise, 60-75°C), kD (maximise, -25 to 50 mL/g), viscosity (minimise, 2-25 cP)
- **Noise levels:** tm=0.013, kd=0.096, visc=0.10
- **Max HV:** 16862 (aggregation pathway), 3588 (oxidation pathway)
- **HV scale:** ~8000-16000 (much larger than coatings)

The mAb domain is the "harder" substrate: higher dimensionality (16D vs 4D), higher noise (CV=49.7% on the aggregation pathway), and — critically — the sigma term DOES change candidate rankings. This means AF structure matters on mAb in a way it doesn't on coatings. The mAb domain is the better evolution substrate because the fitness signal can actually separate AFs.

### Waibel Real mAb Dataset (used as a real-data oracle)

- **Source:** Waibel et al. 2025 [30], 33 real formulations
- **Inputs:** 8 continuous (pH, histidine, sorbitol, arginine, aspartic acid, glutamic acid, acetic acid, HCl)
- **Objectives:** Tm_C (maximise, 63.4-71.3°C), kD_mL_g (maximise, -24.4 to 48.6), RM_Agi_pct (minimise, 1.8-101.1%)
- **Pareto front:** 15/33 = 45.5%
- **Noise-to-signal ratios:** Tm: 0.025 (low), kD: 0.163 (high), RM_Agi: 0.123 (moderate)
- **HV:** ~17,855 (reference point [60, -25, -101] in all-maximise convention)

The Waibel dataset provides real experimental noise (each objective has a confidence interval column). The kD objective has the highest noise-to-signal ratio (0.163), meaning the GP's kD predictions are the least reliable. This dataset was used for the cross-domain transfer test (coatings-evolved AFs applied to mAb) and the 3-seed re-validation.

---

## 4. The Hint Set Evolution

### Design Philosophy

The hint set is the starting gene pool for evolution. Rather than providing many hand-written AF implementations (the original `af_interface.py` had 7 seeds), the v2 design follows FunBO/FunSearch/AlphaEvolve's pattern: **one** hand-written seed (`trust_only` — pure exploitation, the simplest correct implementation) plus a set of one-line **strategy hints** that the LLM is asked to implement at generation 0.

This design tests whether the LLM can generate diversity from descriptions alone, rather than relying on hand-coded variety. In mock mode (no LLM), the evolution has nothing to do with hints — it uses only the seed program plus random-term-weight padding.

### v2.0: The Original 6-Hint Set

The first hint set used in Runs 1, 2, and 3:

| Hint | Mechanism |
|------|-----------|
| `fixed_ucb` | mu_sum + beta * sigma_sum (fixed beta=2.0) |
| `ucb_plus_novelty` | UCB + distance-to-nearest-observed novelty term |
| `phase_decaying_ucb` | UCB with weight decaying as progress increases + stagnation boost |
| `novelty_only` | Pure novelty (distance to nearest observed point) |
| `ehvi_approx` | Expected hypervolume improvement (approximate) |
| `mc_hvi_approx` | Monte Carlo hypervolume improvement |

**Problem discovered:** All 6 hints are UCB-style variants (mu_sum ± w * sigma_sum with different w(progress) schedules). The LLM could not escape UCB space because the hint set offered nothing else. On the mAb run (Run 3), all 8 surviving AFs were simple UCB variants — the evolution explored a tiny corner of AF space.

### v2.1: The Revised 8-Hint Set

After the mAb run showed the LLM was trapped in UCB space, the hint set was revised. Three original hints were kept, three were replaced with upgraded versions of the same mechanism, two were removed, and three genuinely new mechanisms were added:

| # | Hint | Status | Mechanism | Requires |
|---|------|--------|-----------|----------|
| 1 | `fixed_ucb` | Kept | UCB with fixed beta=2.0 | — |
| 2 | `ucb_plus_novelty` | Kept | UCB + input-space novelty | — |
| 3 | `phase_decaying_ucb` | Kept | Progress-decaying UCB + stagnation boost | — |
| 4 | `noisy_front_hvi` | Replaces ehvi_approx + mc_hvi_approx | Resample Pareto front under noise, compute expected HVI | Y_obs, ref_point, pareto_front |
| 5 | `outcome_novelty` | Replaces novelty_only | Objective-space novelty (distance to Y_obs) | Y_obs |
| 6 | `dpp_diversity` | New | DPP-based diversity re-ranking (Nava/Mutný/Krause 2021 [13]) | — |
| 7 | `local_penalization` | New | Greedy local penalization (Gonzalez et al. 2016) | — |
| 8 | `pareto_membership` | New | Pareto-optimality probability via MC (NOSTRA-inspired [1]) | pareto_front |

The goal was 8 hints spanning 8 distinct mechanisms, not 8 variants of UCB. The three new mechanisms (DPP, local penalization, Pareto-membership) were chosen from the literature on batch BO diversity and trust-region methods.

**Removed:** `novelty_only` (pure novelty was dominated by UCB+novelty), `ehvi_approx` and `mc_hvi_approx` (both were approximate HVI without noise handling — replaced by `noisy_front_hvi` which resamples the front under observation noise).

This 8-hint set was used in Run 4. The results are in Section 8 and the bug analysis in Section 9.

---

## 5. Run 1: Coatings, gamma=0.005, n=15

**Run ID:** `run_v2_coatings_real`
**Parameters:** gamma=0.005, 15 training campaigns, 6-hint set, coatings oracle
**LLM:** Real (40 calls, 0 failures)
**Date:** ~July 28, 2026

### Trajectory

The evolution **stagnated at generation 0**. `hint_fixed_ucb` held the lead for all 20 generations with identical fitness:

| Gen | Best Fitness | Best Margin | Best Win Rate | Best HV |
|-----|-------------|-------------|---------------|---------|
| 0 | +0.0543 | +9.43% | 0.733 | 446.25 |
| 1-20 | +0.0543 | +9.43% | 0.733 | 446.25 |

No improvement ever occurred. The best_fitness value never changed across 20 generations.

### Why It Stagnated

The LOC penalty at gamma=0.005 was too aggressive. Consider the top two AFs:

| AF | Margin | LOC | Fitness (margin - 0.005*LOC) |
|----|--------|-----|------|
| hint_fixed_ucb | +9.43% | 8 | +0.0543 |
| gen9_child0 | +9.98% | 10 | +0.0498 |

gen9_child0 had a **higher margin** (+9.98% vs +9.43%) but also higher LOC (10 vs 8). The LOC penalty difference (0.005 × 2 = 0.01) exceeded the margin difference (0.0056), so hint_fixed_ucb won. The evolution could not reward higher-margin AFs that were also more complex.

### Final Population (8 AFs, all UCB-family)

| Rank | AF | Fitness | Margin | LOC | Win Rate | HV |
|------|-----|---------|--------|-----|----------|----|
| 1 | hint_fixed_ucb | +0.0543 | +9.43% | 8 | 0.733 | 446.3 |
| 2 | gen9_child0 | +0.0498 | +9.98% | 10 | 0.800 | 448.8 |
| 3 | trust_only | +0.0494 | +7.94% | 6 | 0.733 | 442.0 |
| 4 | gen6_child1 | +0.0467 | +9.67% | 10 | 0.733 | 448.3 |
| 5 | gen18_child0 | +0.0407 | +9.07% | 10 | 0.667 | 444.9 |
| 6 | gen6_child0 | +0.0373 | +9.23% | 11 | 0.733 | 447.0 |
| 7 | gen4_child1 | +0.0373 | +9.23% | 11 | 0.733 | 447.0 |
| 8 | gen1_child1 | +0.0333 | +8.83% | 11 | 0.733 | 445.4 |

**Structural classification (corrected by user):** 1 fixed UCB, 5 adaptive UCB with progress-decaying weights, 1 pure exploitation (trust_only), 1 random_init. All 8 are UCB-family — the 6-hint set offered nothing else.

**Key AFs:**
- `hint_fixed_ucb`: `mu_sum + 2.0 * sigma_sum` (raw sigma, not normalised). LOC=8, the simplest UCB.
- `gen9_child0`: `w_exploit * mu_sum + (1-w_exploit) * sigma_sum` where `w_exploit = 0.3 + 0.7*(1-progress)`. Blends mu and sigma with a progress-dependent weight. Highest margin (+9.98%) and highest win rate (0.800) but penalised by LOC=10.
- `gen6_child1`: `mu_sum + 2.0*(1-progress) * sigma_sum`. Classic progress-decaying UCB. Weight goes from 2.0 at start to 0.0 at end.
- `gen6_child0`: `mu_sum + (1.0 + 2.0*(1-progress)) * sigma_norm` where `sigma_norm = sum(std/front_range[name])`. Uses front-range-normalised sigma. LOC=11, heavily penalised.

### What This Run Taught Us

1. **gamma=0.005 is too aggressive for coatings.** The LOC penalty overrules margin improvements. The evolution cannot explore higher-complexity AFs.
2. **n=15 campaigns is too few.** The margins are large (~9%) but based on only 15 campaigns — high variance. The n=75 run (Run 2) would show that the true margins are ~3-5%, much smaller.
3. **The LLM produces valid code.** 40 calls, 0 failures. The LLM always generated syntactically correct Python. The problem was not code generation but the fitness landscape.

---

## 6. Run 2: Coatings, gamma=0.005, n=75

**Run ID:** `run_v2_coatings_real_100` (final_population.json in user uploads root)
**Parameters:** gamma=0.005, 75 training campaigns, 6-hint set, coatings oracle
**LLM:** Real (40 calls, 0 failures)
**Date:** ~July 30, 2026

### Trajectory

The evolution **stagnated at generation 0**. `trust_only` (pure exploitation, no uncertainty term) held the lead for all 20 generations:

| Gen | Best Fitness | Best Margin | Best Win Rate | Best HV |
|-----|-------------|-------------|---------------|---------|
| 0 | +0.0038 | +3.38% | 0.680 | 436.4 |
| 1-20 | +0.0038 | +3.38% | 0.680 | 436.4 |

Note the dramatic difference from Run 1: with 75 campaigns instead of 15, the margins shrank from ~9% to ~3.4%. The n=15 run's large margins were an artefact of small-sample variance.

### Why trust_only Won (Not hint_fixed_ucb)

With 75 campaigns, the margin magnitudes are smaller and more stable. trust_only (LOC=6) has the lowest LOC in the population. Its margin (+3.38%) is lower than gen3_child0's (+4.39%), but the LOC penalty makes up the difference:

| AF | Margin | LOC | Fitness |
|----|--------|-----|---------|
| trust_only | +3.38% | 6 | +0.0038 |
| hint_fixed_ucb | +4.67% | 8 | +0.0067 |
| gen3_child0 | +4.39% | 10 | -0.0061 |

Wait — hint_fixed_ucb actually has higher fitness (+0.0067) than trust_only (+0.0038). But trust_only won the evolution. This is because the evolution's selection is based on the best fitness at each generation, and trust_only was the initial seed that the population was built around. The history shows trust_only as best at gen 0 with fitness +0.0038, and this never changed. The hint_fixed_ucb with fitness +0.0067 must have been generated later but the history records the gen-0 best as trust_only.

Actually, looking more carefully: the final population shows hint_fixed_ucb with fitness +0.0067 (rank 1) and trust_only with +0.0038 (rank 2). But the history shows best_fitness = +0.0038 for all 20 generations. This means trust_only was the gen-0 best, and hint_fixed_ucb was already in the initial population (it's a hint, not an LLM-generated AF) but was not selected as the "best" at gen 0 — possibly because the initial evaluation used a different seed or the selection logic picks the first AF in case of ties. The history's best_fitness tracks the running maximum, and if hint_fixed_ucb was evaluated at gen 0 with fitness +0.0067, it should have been recorded as best. The discrepancy suggests the initial population evaluation may have used different campaign seeds than the final population evaluation, or the history records only the LLM-generated offspring's best, not the seed/hint AFs.

**Correction:** On re-examination, the final population is the surviving population at gen 20, which includes both seeds and LLM-generated AFs. The history's best_fitness at gen 0 is +0.0038 (trust_only), but the final population shows hint_fixed_ucb at +0.0067. This means hint_fixed_ucb was in the initial population but was NOT the best at gen 0 — which is only possible if the initial evaluation used different random seeds than the final population's recorded metrics. The n_campaigns=75 in the final population confirms these are the full 75-campaign evaluations. The history likely records the best among LLM-generated offspring only, or there was a re-evaluation between gen 0 and the final population dump.

Regardless of this bookkeeping detail, the key finding stands: **the evolution stagnated at gen 0 and no LLM-generated AF ever beat the initial population.**

### Final Population (8 AFs, all UCB-family)

| Rank | AF | Fitness | Margin | LOC | Win Rate | HV |
|------|-----|---------|--------|-----|----------|----|
| 1 | hint_fixed_ucb | +0.0067 | +4.67% | 8 | 0.640 | 442.1 |
| 2 | trust_only | +0.0038 | +3.38% | 6 | 0.680 | 436.4 |
| 3 | gen3_child0 | -0.0061 | +4.39% | 10 | 0.787 | 440.8 |
| 4 | gen4_child0 | -0.0090 | +4.10% | 10 | 0.707 | 439.5 |
| 5 | gen11_child1 | -0.0097 | +4.03% | 10 | 0.760 | 439.2 |
| 6 | gen20_child0 | -0.0110 | +4.40% | 11 | 0.760 | 441.0 |
| 7 | gen12_child1 | -0.0132 | +3.68% | 10 | 0.667 | 438.0 |
| 8 | gen10_child1 | -0.0142 | +4.08% | 11 | 0.693 | 439.4 |

Note: 6 of 8 AFs have **negative fitness** — their margin doesn't cover the LOC penalty. Only the two lowest-LOC AFs (hint_fixed_ucb at LOC=8 and trust_only at LOC=6) have positive fitness.

**Best by raw margin:** gen3_child0 at +4.39% (win rate 0.787, the highest in the population) but LOC=10 makes its fitness negative. gen20_child0 has margin +4.40% but LOC=11.

### Held-Out Validation (25 campaigns)

The final population was validated on 25 held-out coatings campaigns (not used in training). Results reported by the user:

- **5 of 8 AFs significant** (p < 0.05, Wilcoxon signed-rank test)
- Margins ranged from ~+2.9% to ~+6.4%
- The significant winners were the higher-margin AFs (gen3_child0, gen20_child0, etc.) — the held-out validation doesn't apply the LOC penalty, so the raw margin is what matters

This was initially exciting (5/8 significant!) but the cross-domain transfer test (Section 11) and the noise diagnostic (Section 12) would later show that much of this signal was seed noise, not genuine AF quality.

### What This Run Taught Us

1. **n=15 inflated margins.** The true coatings margins are ~3-5%, not ~9%. More campaigns are needed for stable fitness estimates.
2. **gamma=0.005 still too aggressive.** 6/8 AFs have negative fitness. The LOC penalty prevents the evolution from selecting the best-margin AFs.
3. **The best-margin AF (gen3_child0, +4.39%) has the highest win rate (0.787).** The evolution is finding good AFs but the fitness function is not selecting them.
4. **5/8 significant on held-out is promising but needs noise checking.** The significance could be driven by seed variance, not genuine AF superiority.

---

## 7. Run 3: mAb, gamma=0.005, n=75

**Run ID:** `run_v2_mAb_real_100`
**Parameters:** gamma=0.005, 75 training campaigns, 6-hint set, mAb excipient oracle
**LLM:** Real (40 calls, 0 failures)
**Date:** ~July 30, 2026

### Trajectory

The evolution **stagnated at generation 0**. `trust_only` held the lead for all 20 generations:

| Gen | Best Fitness | Best Margin | Best Win Rate | Best HV |
|-----|-------------|-------------|---------------|---------|
| 0 | +0.0124 | +4.24% | 0.587 | 8154.2 |
| 1-20 | +0.0124 | +4.24% | 0.587 | 8154.2 |

### Final Population (8 AFs, all UCB-family)

| Rank | AF | Fitness | Margin | LOC | Win Rate | HV |
|------|-----|---------|--------|-----|----------|----|
| 1 | trust_only | +0.0124 | +4.24% | 6 | 0.587 | 8154.2 |
| 2 | gen15_child1 | +0.0037 | +5.37% | 10 | 0.627 | 8302.2 |
| 3 | gen6_child1 | -0.0153 | +3.47% | 10 | 0.507 | 8155.9 |
| 4 | random_init_0 | -0.0176 | +4.24% | 12 | 0.587 | 8154.2 |
| 5 | hint_fixed_ucb | -0.0183 | +2.17% | 8 | 0.493 | 8242.9 |
| 6 | gen3_child1 | -0.0186 | +3.14% | 10 | 0.507 | 8113.5 |
| 7 | gen18_child0 | -0.0208 | +2.42% | 9 | 0.480 | 8099.8 |
| 8 | gen11_child1 | -0.0211 | +2.89% | 10 | 0.507 | 8133.2 |

**6 of 8 AFs have negative fitness.** Only trust_only and gen15_child1 have positive fitness.

### Why mAb Evolution Cannot Work (Yet)

The mAb excipient oracle has a **noise floor** that makes the fitness signal too noisy to distinguish AFs:

- **CV = 49.7%** (coefficient of variation of HV across campaigns)
- **SE = 0.057** (standard error of the mean margin)
- **Signal gap = 0.011** (the margin difference between the best and worst AF)

The standard error (0.057) is **5× larger** than the signal gap (0.011). The evolution is optimising against noise, not signal. Any AF that "wins" does so because of lucky seed draws, not because it's genuinely better.

This was confirmed by the noise diagnostic (Section 12): 29% of total variance comes from pipeline seed noise (GP-fit seed + U-NSGA-III seed), and 71% comes from domain noise (the oracle's stochasticity). Even after fixing the seed noise, the SE drops from 0.057 to 0.048 — still far above the 0.011 signal gap.

### The random_init_0 Anomaly

`random_init_0` is a randomly-generated AF (not LLM-generated, not a hint). Its code is:

```python
def score_pool(context):
    """weighted sum of: mu_sum(4.09)"""
    s = 0.0
    s = s + (4.0850) * (sum(gp[name]['mean'] for name in names))
    scores.append(s)
```

This is just `4.085 * mu_sum` — pure exploitation with a scalar multiplier. Since `score_pool` returns scores used for **ranking** (not absolute values), the scalar multiplier doesn't change the ranking at all. `random_init_0` is rank-equivalent to `trust_only`. Indeed, both have identical margin (+4.24%), win rate (0.587), and HV (8154.2) — confirming they produce the same candidate selections.

### What This Run Taught Us

1. **mAb evolution is blocked by noise.** The fitness signal (0.011) is 5× smaller than the noise (SE=0.057). No amount of evolution can find signal in noise.
2. **The 6-hint set traps the LLM in UCB space.** All 8 surviving AFs are UCB variants. The LLM needs genuinely different mechanisms to explore (hence the v2.1 8-hint set).
3. **trust_only wins on mAb too.** Pure exploitation is the safest strategy when the fitness is noisy — it has the lowest LOC and a reasonable margin.
4. **gen15_child1 has the highest margin (+5.37%) but is penalised by LOC=10.** If gamma were lower, it would win. This foreshadows the gamma=0.001 recalibration.

---

## 8. Run 4: Coatings, gamma=0.001, n=75, 8-Hint Set

**Run ID:** `run_v2_coatings_gamma001_v2`
**Parameters:** gamma=0.001, 75 training campaigns, 8-hint set (v2.1), coatings oracle
**LLM:** Real (40 calls, 0 failures)
**Date:** ~July 31, 2026

### Trajectory

This run **progressed** — the first and only run where an LLM-generated AF beat the initial population:

| Gen | Best AF | Best Fitness | Best Margin | Best Win Rate | Best HV |
|-----|---------|-------------|-------------|---------------|---------|
| 0 | trust_only | +0.0278 | +3.378% | 0.680 | 436.4 |
| 1-5 | trust_only | +0.0278 | +3.378% | 0.680 | 436.4 |
| **6** | **gen6_child0** | **+0.0284** | **+3.745%** | **0.693** | **438.0** |
| 7-20 | gen6_child0 | +0.0284 | +3.745% | 0.693 | 438.0 |

At generation 6, gen6_child0 beat trust_only. The margin improvement (+3.745% vs +3.378% = +0.367%) exceeded the LOC penalty difference (0.001 × 3 = 0.003, since gen6_child0 has LOC=9 vs trust_only's LOC=6). This is the gamma=0.001 recalibration working as intended.

After gen 6, the evolution stagnated for 14 more generations. gen6_child0 held the lead with no further improvement.

### The Winner: gen6_child0

```python
def score_pool(context):
    """Exploitation with uncertainty bonus: sum of means plus a scaled
    uncertainty term to encourage exploration near the Pareto front."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        scores.append(mu_sum + 0.5 * sigma_norm)
    return scores
```

This is a UCB variant with:
- **Front-range-normalised sigma:** `sigma_norm = sum(std / front_range[name])` — normalises each objective's uncertainty by the range of the current Pareto front, making the uncertainty term scale-invariant across objectives.
- **Fixed weight 0.5:** A modest uncertainty bonus. Much smaller than hint_fixed_ucb's beta=2.0.
- **No progress dependence:** Unlike most other AFs in the population, gen6_child0 does not use `context["campaign"]["progress"]`. The weight is constant throughout the campaign.

### Final Population (8 AFs, all UCB-family)

| Rank | AF | Fitness | Margin | Median Margin | LOC | Win Rate | HV |
|------|-----|---------|--------|---------------|-----|----------|----|
| 1 | gen6_child0 | +0.0284 | +3.745% | +0.181% | 9 | 0.693 | 438.0 |
| 2 | trust_only | +0.0278 | +3.378% | +0.097% | 6 | 0.680 | 436.4 |
| 3 | hint_fixed_ucb | +0.0274 | +3.645% | +0.148% | 9 | 0.693 | 437.6 |
| 4 | gen4_child0 | +0.0270 | +4.097% | +0.292% | 14 | 0.693 | 439.5 |
| 5 | gen1_child1 | +0.0268 | +4.078% | +0.292% | 14 | 0.693 | 439.4 |
| 6 | gen9_child0 | +0.0266 | +3.759% | +0.181% | 11 | 0.693 | 438.1 |
| 7 | gen5_child0 | +0.0264 | +3.740% | +0.181% | 11 | 0.693 | 438.0 |
| 8 | gen10_child1 | +0.0262 | +3.721% | +0.181% | 11 | 0.680 | 437.8 |

Note: this run introduced `median_margin` as an additional recorded statistic (though fitness still uses mean margin). The median margins are much smaller than the mean margins (0.18% vs 3.7%), indicating the mean is inflated by outlier campaigns. This observation would later drive the proposal to use median-based fitness.

### Structural Classification (Corrected by User)

The user corrected the assistant's initial structural classification. The correct breakdown is **3/3/1** (not 4/2/1):

- **3 adaptive UCB** (progress-decaying weight): gen4_child0, gen1_child1, gen9_child0, gen5_child0 — wait, that's 4. The user's correction was more nuanced:
  - 3 AFs with progress-decaying uncertainty weight (gen5_child0, gen9_child0, gen10_child1)
  - 3 AFs with progress + stagnation terms (gen4_child0, gen1_child1 — these add a `novelty_factor = 1.0 + min(stagnant_batches, N) * coefficient`)
  - 1 fixed UCB (hint_fixed_ucb, beta=2.0 with normalised sigma)
  - 1 pure exploitation (trust_only)

The user also corrected that `trust_only` breaks the adaptive-vs-fixed dichotomy — it's neither, it's pure exploitation with no uncertainty term at all.

### The Critical Finding: NONE of the 4 New Mechanisms Survived

Despite the 8-hint set offering DPP diversity, local penalization, noisy-front HVI, and Pareto-membership, **none of these mechanisms appear in the final population.** All 8 surviving AFs are UCB-family. The 4 new mechanisms were either:

1. **Broken** (noisy_front_hvi, local_penalization) — produced buggy code that fell back to random selection
2. **Drifted** (pareto_membership) — produced working code that measured the wrong thing
3. **Correct but lost** (dpp_diversity) — produced correct code but was outcompeted by mu_sum dominance

This means Run 4 is **NOT a clean test** of whether the new mechanisms add value on coatings. The only clean signal is dpp_diversity (correct code, lost to UCB) — which suggests mu_sum dominance on coatings means no mechanism can beat pure exploitation there. The bug analysis is in Section 9.

### Held-Out Validation (25 campaigns)

The user reported **8/8 significant** (p < 0.05) on the held-out coatings validation. However, there was a critical caveat:

**7 of 8 AFs had IDENTICAL held-out HV = 444.8 and identical p = 0.0160.** Only trust_only differed (HV = 443.7, p = 0.0187). This means 7 of 8 AFs are **rank-equivalent** on coatings — they produce the same candidate selections despite having different code. Any nonzero uncertainty weight (fixed, progress-decaying, or stagnation-boosted) produces the same rankings.

The 6.4% improvement over EGBO-novelty comes from **mu_sum exploitation itself**, not the sigma weight. The sigma term's marginal contribution is only ~0.3% (443.7 → 444.8, the difference between trust_only and everything else).

This is the **mu_sum dominance** finding: on coatings, the GP's mean prediction is good enough that any reasonable uncertainty bonus produces the same batch selections. The AF structure doesn't matter — only whether you include sigma at all (trust_only vs everything else).

### What This Run Taught Us

1. **gamma=0.001 works for coatings.** The evolution progressed (gen6_child0 beat trust_only at gen 6). The margin improvement can overcome the LOC penalty.
2. **mu_sum dominance on coatings.** 7/8 AFs are rank-equivalent. The sigma term doesn't change rankings. The evolution is optimising something that doesn't matter on this domain.
3. **The 4 new mechanisms didn't survive.** 2 were broken, 1 drifted, 1 was correct but lost. The 8-hint set did not diversify the final population.
4. **Median margin << mean margin.** The mean is inflated by outliers. This motivates median-based fitness.
5. **Coatings is the wrong substrate for AF evolution.** The evolution can't optimise what it can't distinguish. mAb (where AFs are distinguishable) is the right substrate, but mAb needs noise-robust fitness first.

---

## 9. The Four New Hints: Bug Analysis

After Run 4, the user manually inspected the 4 LLM-generated implementations of the new hints (from the `af_code_logs/` directory). The assistant then verified the bugs by reproducing them in Python. Here is the complete analysis:

### 9.1 dpp_diversity (call_00006) — CORRECT but Lost

```python
def score_pool(context):
    """Rank candidates by normalized predicted quality multiplied by
    diversity from high-scoring peers."""
    names = context["objective_names"]
    pool = context["pool"]
    qualities = np.array([sum(cand["gp_posterior"][name]["mean"]
                              for name in names) for cand in pool])
    q_min, q_max = qualities.min(), qualities.max()
    if q_max > q_min:
        norm_qualities = (qualities - q_min) / (q_max - q_min + 1e-9)
    else:
        norm_qualities = np.zeros_like(qualities)
    X = np.array([cand["x"] for cand in pool])
    distances = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(distances, np.inf)
    similarity = np.exp(-distances)
    n_top = max(1, len(pool) // 10)
    top_indices = np.argsort(norm_qualities)[-n_top:]
    diversities = []
    for i in range(len(pool)):
        sim_to_top = similarity[i, top_indices]
        diversity = np.mean(sim_to_top)
        diversities.append(diversity)
    scores = norm_qualities * (1 - np.array(diversities))
    return scores.tolist()
```

**Assessment:** Correct implementation. Min-max normalises quality to [0,1], builds a vectorised similarity matrix, excludes self-similarity (diagonal set to inf), computes diversity as mean similarity to top-10% candidates, and combines as `quality * (1 - diversity)`.

**Why it lost:** mu_sum dominance on coatings. The quality term (`norm_qualities`) is a normalised version of mu_sum, and the diversity penalty (`1 - diversity`) is a secondary effect. On coatings, where any uncertainty weight produces the same rankings, the diversity term doesn't change the outcome enough to beat pure UCB. This is the **one clean signal** from the 4 new hints: a correct, non-UCB mechanism that lost to mu_sum dominance.

### 9.2 noisy_front_hvi (call_00004) — BROKEN: Repetition-Loop Pathology

The file is 71 lines long. **54 of 71 lines are comments.** Only 16 lines are actual code. The function never reaches a `return` statement — it burns its entire token budget writing increasingly desperate comments about how it should simplify the hypervolume computation.

The comment repetition pattern (20 hedging phrases: "placeholder", "simplify", "For now", "not correct", "not exact", "avoid complexity") shows the LLM got stuck in a loop of:
1. Starting to write a HV computation
2. Deciding it's too complex
3. Saying "let's simplify"
4. Starting again with a different simplification
5. Deciding that's also too complex
6. Repeating

The file ends mid-comment: `"# is the minimum in each dimension"` — the LLM ran out of tokens without ever returning scores. The `scores` array (initialised as `np.zeros(len(pool))`) is never populated, so the function returns all zeros → all candidates tie → random selection.

**Root cause:** The hint asks the LLM to implement hypervolume improvement computation, which is genuinely complex (especially for >2 objectives). The LLM knows it can't implement a correct HV algorithm from scratch in a single function, and instead of writing a simple approximation, it spirals into commentary about why it can't do it.

**Fix proposed:** Reword the hint to say "write code immediately, minimal comments, no extended commentary" and suggest a simple proxy (e.g. distance from candidate to nearest front point in objective space) rather than full HV computation.

### 9.3 local_penalization (call_00007) — BROKEN: 3 Bugs

The LLM produced code with three distinct bugs, all verified by Python reproduction:

**Bug 1: IndexError (scores list never initialised)**

```python
scores = []  # Empty list
# ... later:
scores[idx] = base_scores[idx] * (1.0 - penalty_factor)  # IndexError!
```

`scores` is initialised as an empty list, then the code tries to assign to `scores[idx]` by index. In Python, you can't assign to an index that doesn't exist in a list. This crashes on the first iteration.

**Bug 2: Inverted penalty direction**

```python
penalty_factor = 1.0 - np.exp(-radius)
scores[idx] = base_scores[idx] * (1.0 - penalty_factor)
# = base_scores[idx] * np.exp(-radius)
```

For a **close** candidate (radius → 0): `exp(-radius) → 1.0` → score barely reduced (multiplier ≈ 1.0)
For a **far** candidate (radius → ∞): `exp(-radius) → 0.0` → score heavily reduced (multiplier ≈ 0.0)

This is **backwards.** Close candidates (near an already-picked one) should be penalised MORE (lower score), and far candidates should be penalised LESS (higher score). The code does the opposite: it penalises far candidates and leaves close ones alone.

Verified numerically:
- Close (radius=0.01): multiplier = 0.99 (almost no penalty)
- Far (radius=5.0): multiplier = 0.007 (almost total penalty)

**Bug 3: TypeError (float used as list index)**

```python
best_idx = remaining[base_scores[remaining[0]] if len(remaining) == 1
                     else max(remaining, key=lambda i: base_scores[i])]
```

When `len(remaining) == 1`, the expression `base_scores[remaining[0]]` evaluates to a **float** (e.g. 0.8), and then `remaining[0.8]` tries to index a list with a float → TypeError.

**Fix proposed:** Reword the hint to (1) say "initialise scores to base scores, not empty list", (2) make the suppression direction explicit ("multiply by suppression factor, not by 1-penalty"), (3) simplify the best-candidate selection logic.

### 9.4 pareto_membership (call_00008) — DRIFTED: Wrong Dominance Direction

The code is syntactically correct and runs without errors. The performance is reasonable (it precomputes MC samples once, which is efficient). But it checks the **wrong dominance relationship.**

**What the code checks:** "Does the sample DOMINATE a front point?"
```python
n_dominated = sum(1 for s in samples if not any(
    all(s[i] >= pf_point[i] for i in range(len(names))) and
    any(s[i] > pf_point[i] for i in range(len(names)))
    for pf_point in pf
))
pareto_probs.append(1.0 - n_dominated / n_samples)
```

The inner `any(...)` checks whether the sample dominates any Pareto front point. If it does, `n_dominated` is NOT incremented (the `not any(...)` is True). If the sample doesn't dominate any front point, `n_dominated` IS incremented. So `pareto_prob = 1 - n_dominated/n_samples` measures "probability that the sample dominates at least one front point."

**What it should check:** "Is the sample DOMINATED BY a front point?"

The hint says "estimate each candidate's probability of being Pareto-optimal" — a candidate is Pareto-optimal if it is NOT dominated by any existing front point. The code instead measures whether the candidate dominates a front point (i.e. improves the front), which is a different event.

**Verified with a concrete example:**
- Sample = [0.85, 0.35], Pareto front = {[0.8, 0.3], [0.6, 0.5]}
- Sample dominates pf[0] (0.85≥0.8 and 0.35≥0.3, with 0.85>0.8)
- Sample is NOT dominated by any front point (pf[0]: 0.8<0.85; pf[1]: 0.6<0.85)
- Code gives: pareto_prob = 1 - 1/25 = 0.96 (penalised for dominating the front)
- Correct: pareto_prob = 1 - 0/25 = 1.0 (not dominated, so Pareto-optimal)

The code **penalises candidates that improve the front** (dominate a front point) instead of candidates that are **already covered** (dominated by the front). This is the opposite of what a Pareto-membership AF should do.

**Fix proposed:** Reword the hint to be explicit about dominance direction: "check whether each sample is DOMINATED BY any point in the Pareto front (front point >= sample in all objectives), not whether the sample dominates a front point."

### Summary of the 4 New Hints

| Hint | Status | Outcome | Clean Signal? |
|------|--------|---------|---------------|
| dpp_diversity | Correct | Lost to mu_sum dominance | Yes — non-UCB mechanism can't beat exploitation on coatings |
| noisy_front_hvi | Broken (repetition loop) | Returns all zeros → random | No — never tested |
| local_penalization | Broken (3 bugs) | Crashes → fallback | No — never tested |
| pareto_membership | Drifted (wrong direction) | Penalises front-improving candidates | No — measures wrong event |

**Bottom line:** 2/4 broken, 1/4 drifted, 1/4 correct-but-lost. Run 4 is NOT a clean test of whether new mechanisms add value. The only clean data point is dpp_diversity losing, which supports the mu_sum dominance finding. The 3 broken/drifted hints need to be repaired before any conclusion about mechanism diversity can be drawn.

---

## 10. Held-Out Validation Results

### Coatings Held-Out (Run 2 population, 25 campaigns)

The 8 AFs from Run 2 (gamma=0.005, n=75) were validated on 25 held-out coatings campaigns:

- **5 of 8 AFs significant** (p < 0.05, Wilcoxon signed-rank)
- Margins: ~+2.9% to ~+6.4%
- The significant winners were the higher-margin AFs (gen3_child0, gen20_child0, etc.)
- The held-out validation does NOT apply the LOC penalty, so raw margin determines significance

The user initially retracted the "hint_fixed_ucb overfits" narrative — it was a false negative at n=5 held-out campaigns, not a genuine overfitting signal. With more held-out campaigns, hint_fixed_ucb was significant.

### Coatings Held-Out (Run 4 population, 25 campaigns)

The 8 AFs from Run 4 (gamma=0.001, n=75, 8-hint) were validated on 25 held-out coatings campaigns:

- **8 of 8 AFs significant** (p < 0.05)
- **BUT 7 of 8 had IDENTICAL HV = 444.8 and identical p = 0.0160** — rank-equivalent
- Only trust_only differed (HV = 443.7, p = 0.0187)
- The 6.4% improvement over EGBO-novelty comes from mu_sum exploitation, not the sigma weight
- The sigma term's marginal contribution: ~0.3% (443.7 → 444.8)

### mAb Held-Out (Run 2 coatings-evolved AFs, 20 campaigns, single-seed)

The 8 coatings-evolved AFs from Run 2 were tested on 20 mAb campaigns (cross-domain transfer):

- **0 of 8 AFs significant** (p < 0.05)
- All p-values > 0.05 — no transfer from coatings to mAb

This was the first indication that coatings-evolved AFs are overfit to the coatings domain.

### mAb Held-Out (Run 4 coatings-evolved AFs, 20 campaigns, single-seed)

The 8 coatings-evolved AFs from Run 4 were tested on 20 mAb campaigns:

- **4 of 8 AFs significant** (p < 0.05) — initially reported
- AFs are NOT rank-equivalent on mAb (188 HV spread between best and worst)
- The sigma term DOES change rankings on mAb (unlike coatings)
- trust_only is consistently non-significant on mAb

### mAb Held-Out (Run 4 AFs, 20 campaigns, 3-seed averaged) — see Section 13

- **3 of 8 AFs significant** (down from 4/8 single-seed)
- The hint_fixed_ucb flip (p=0.596 → p=0.030) proved seed noise was inflating/dflating significance

---

## 11. Cross-Domain Transfer Tests

### Coatings → mAb (Run 2 population)

- **Setup:** 8 coatings-evolved AFs from Run 2, tested on 20 mAb excipient oracle campaigns
- **Result:** 0/8 significant (p > 0.05 for all)
- **Interpretation:** Coatings-evolved AFs do not transfer to mAb. The AFs are overfit to the coatings domain's specific landscape (4D, near-independent objectives, mu_sum dominance).

### Coatings → Synthetic Benchmarks (Run 2 population)

- **Setup:** 8 coatings-evolved AFs from Run 2, tested on ZDT1 (2D, 2 objectives) and DTLZ2 (10D, 5 objectives), 20 campaigns each
- **Baseline:** mo_egbo_novelty on each synthetic domain

**ZDT1 results (verified from data files):**

| AF | Mean Margin | p-value | Significant? |
|----|-------------|---------|--------------|
| hint_fixed_ucb | -0.29% | 0.294 | No |
| trust_only | -0.76% | 0.024 | Yes (negative) |
| gen3_child0 | -0.79% | 0.010 | Yes (negative) |
| gen4_child0 | -0.44% | 0.105 | No |
| gen11_child1 | -0.10% | 0.674 | No |
| gen20_child0 | -0.58% | 0.027 | Yes (negative) |
| gen12_child1 | -0.52% | 0.058 | No |
| gen10_child1 | -0.65% | 0.022 | Yes (negative) |

**0/8 positive significant.** 4/8 are significantly WORSE than the baseline. The coatings-evolved AFs hurt performance on ZDT1.

**DTLZ2 results (verified from data files):**

| AF | Mean Margin | p-value | Significant? |
|----|-------------|---------|--------------|
| hint_fixed_ucb | +0.54% | 0.956 | No |
| trust_only | -3.06% | 0.019 | Yes (negative) |
| gen3_child0 | -0.85% | 0.231 | No |
| gen4_child0 | -1.11% | 0.177 | No |
| gen11_child1 | +0.22% | 0.571 | No |
| gen20_child0 | -0.14% | 0.927 | No |
| gen12_child1 | +0.35% | 0.674 | No |
| gen10_child1 | -0.84% | 0.277 | No |

**0/8 positive significant.** Only trust_only is significantly worse. The rest are indistinguishable from baseline.

**Interpretation:** Coatings-evolved AFs do not generalise to synthetic benchmarks. The UCB-style AFs evolved on coatings (where mu_sum dominance holds) are overfit to that domain's specific landscape. On ZDT1 (noiseless, smooth, convex Pareto front), the uncertainty bonus hurts. On DTLZ2 (noiseless, 5 objectives), the AFs are indistinguishable from baseline.

This is the **floor rule** in action: on noiseless synthetic domains, the GP's posterior is so accurate that any uncertainty bonus is wasted — the GP already knows where the optimum is. The AF structure that helps on noisy real-data domains (coatings, mAb) is counterproductive on noiseless domains.

### Key Insight: The Floor Rule (Domain-Specific)

The "floor rule" is a pattern observed across domains:

- **On noisy real-data domains (coatings, mAb):** Adding any uncertainty bonus (sigma term) to mu_sum improves HV by ~3-6% over pure exploitation. The GP's mean predictions are unreliable enough that exploration helps.
- **On noiseless synthetic domains (ZDT1, DTLZ2):** Adding an uncertainty bonus hurts or has no effect. The GP's mean predictions are accurate enough that exploration is wasted.

**Important caveat (user correction):** This is a **re-description of observed data**, not yet a testable causal rule. The noise-sweep (varying observation noise on ZDT1/DTLZ2 to see if the floor rule reverses) is still pending. Without the noise-sweep, we cannot distinguish "noise causes the floor rule" from "dimensionality/landscape structure causes it."

---

## 12. The Noise Diagnostic

### Motivation

The user noticed non-determinism in the 20-campaign comparison results: running the same AF on the same campaign with different random seeds produced different HV values. Rather than chasing determinism (fixing all seeds), the user proposed measuring the variance directly via multiple replicates.

### Setup

- **Fixed AF:** gen6_child0 (the Run 4 winner)
- **Fixed campaign:** agg0.25_seed005 (a specific mAb aggregation campaign)
- **Varied:** GP-fit seed (0-14) and U-NSGA-III seed (0-14), independently
- **15 replicates per axis**

### Results

**GP-fit seed axis (U-NSGA-III seed fixed):**
- Mean HV: 3000
- Std: 443
- CV: 14.78%

**U-NSGA-III seed axis (GP-fit seed fixed):**
- Mean HV: 3227
- Std: 418
- CV: 12.94%

**Joint-seed paired margin** (both seeds varied, AF vs baseline paired):
- Mean: +0.035 (3.5%)
- Std: 0.256 (25.6%)
- The margin swings from -31% to +79% depending on seeds

### Variance Decomposition

- **Seed noise (GP-fit + U-NSGA-III):** 29% of total variance
- **Domain noise (oracle stochasticity):** 71% of total variance

### The Pairing Failure

A natural idea: pair the AF and baseline on the same seed so they see the same random numbers, and the seed noise cancels in the difference. **This does not work.**

The AF and baseline diverge from batch 1 onward: they select different candidates → query different points → train on different data → their GP models diverge → their RNG streams decorrelate. By batch 2-3, the AF and baseline are effectively running on different random number streams despite starting with the same seed. The pairing cancels the initial-design noise but not the downstream pipeline noise.

### Implications

Two fixes are needed, addressing different noise sources:

1. **Multi-seed averaging (cuts 29%):** Run each campaign evaluation with multiple GP-fit and U-NSGA-III seeds, average the HV. This reduces the seed noise by √n (3 seeds → ~√3 reduction). The 3-seed re-validation (Section 13) confirmed this works.

2. **Median/bootstrap fitness (addresses 71%):** The domain noise comes from the oracle's stochasticity — different campaigns have different landscapes, and the margin distribution across campaigns is wide (std 0.494) and skewed (swings -0.31 to +0.79). The mean margin is inflated by outlier campaigns. Using the median margin (or a bootstrap confidence interval) instead of the mean would make the fitness more robust to these outliers.

**Even after both fixes, the signal may still be thin.** The SE drops from 0.057 to ~0.048 after seed averaging — still much larger than the 0.011 signal gap on mAb. The median helps with outliers but doesn't reduce the fundamental variance. The mAb evolution may need more campaigns (n>75) or more seeds (n_seeds>3) to achieve a stable fitness signal.

---

## 13. The 3-Seed mAb Re-Validation

### Setup

- Same 8 coatings-evolved AFs from Run 4
- Same 20 mAb campaigns
- Averaged over 3 seeds per campaign (instead of 1)
- Question: does seed averaging change which AFs are significant?

### Results

| AF | 1-seed p | 3-seed p | Verdict |
|----|----------|----------|---------|
| gen5_child0 (progress-decaying UCB, beta=2.0*(1-progress)) | 0.040 | 0.017 | HELD, strengthened |
| gen9_child0 (progress-decaying UCB, beta=1.5*(1-progress²)) | 0.040 | 0.027 | HELD, strengthened |
| gen4_child0 (progress + stagnation UCB) | 0.048 | 0.053 | DROPPED below 0.05 |
| gen1_child1 (progress + stagnation UCB) | 0.048 | 0.053 | DROPPED below 0.05 |
| hint_fixed_ucb (fixed UCB, beta=2.0) | 0.596 | 0.030 | NEW — was masked by seed noise |
| gen6_child0 (fixed UCB, beta=0.5, normalised sigma) | 0.053 | 0.090 | Never significant |
| gen10_child1 (adaptive UCB, 0.5+0.3*(1-progress)) | 0.053 | 0.090 | Never significant |
| trust_only (pure exploitation, no sigma) | 0.294 | 0.083 | Consistently not significant |

### The hint_fixed_ucb Flip: The Single Most Important Data Point

hint_fixed_ucb went from p=0.596 (completely non-significant) to p=0.030 (significant) under seed averaging. Same AF, same campaigns, same oracle — only the number of seeds changed.

This is the **empirical confirmation of the noise diagnostic.** A single-seed evaluation would have discarded this AF as unremarkable. The evolution was optimising against seed noise, not signal. Any AF that "won" or "lost" in a single-seed evaluation could be a seed artefact.

### Structural Pattern

The significant AFs under seed averaging share a structural family:

- **gen5_child0 and gen9_child0:** Simple progress-decaying UCB (beta decreases as progress increases). No stagnation term. These HELD and STRENGTHENED under seed averaging.
- **gen4_child0 and gen1_child1:** Progress-decaying UCB + stagnation novelty boost. These DROPPED below significance under seed averaging.
- **hint_fixed_ucb:** Fixed UCB (no progress dependence). Was masked by seed noise, revealed by averaging.

**The stagnation term is a liability.** It adds a second source of variability (depends on `stagnant_batches`, which is itself noisy — whether the HV improved in the last batch depends on the seed). The simpler progress-only decay (gen5/gen9) is more robust because it has fewer noisy inputs.

**trust_only is consistently non-significant on mAb** across both single-seed and 3-seed evaluations. The sigma term matters on mAb (unlike coatings where trust_only was rank-equivalent to everything). Pure exploitation is not enough on mAb — you need some uncertainty bonus.

### Verdict

**Real-but-thin effect.** 3/8 significant after noise reduction (vs 4/8 before). The p-values cluster at 0.017-0.09, not the clean 8/8 picture coatings gave. The signal is genuine (it strengthens under noise reduction for the robust AFs) but small. Proceed to mAb evolution with n_fitness_seeds ≥ 3.

---

## 14. Synthetic Benchmark Generalization

### Setup

The 8 coatings-evolved AFs from Run 2 (gamma=0.005, n=75) were tested on two standard MOO benchmark domains:

- **ZDT1:** 2D continuous, 2 objectives (both minimised), convex Pareto front, noiseless
- **DTLZ2:** 10D continuous, 5 objectives (all minimised), spherical Pareto front, noiseless

20 campaigns per domain, budget=40, n_init=10, batch_size=5. Baseline: mo_egbo_novelty on each domain.

### Results (Verified from Data Files)

**ZDT1 (2D, 2 objectives):**
- 0/8 AFs significantly better than baseline
- 4/8 AFs significantly WORSE (trust_only p=0.024, gen3_child0 p=0.010, gen20_child0 p=0.027, gen10_child1 p=0.022)
- Mean margins all negative or near-zero (-0.10% to -0.79%)
- The UCB-style AFs evolved on coatings actively hurt on ZDT1

**DTLZ2 (10D, 5 objectives):**
- 0/8 AFs significantly better than baseline
- 1/8 significantly worse (trust_only p=0.019)
- Most AFs indistinguishable from baseline (p > 0.17)
- Mean margins range from -3.06% to +0.54%

### Interpretation

Coatings-evolved AFs do not generalise to synthetic benchmarks. The AFs are overfit to the coatings domain. On noiseless synthetic domains, the GP's posterior is so accurate that the uncertainty bonus is wasted or counterproductive.

This is consistent with the floor rule: the uncertainty term helps on noisy real-data domains (where the GP's mean is unreliable) but hurts on noiseless synthetic domains (where the GP's mean is accurate). The AFs evolved on coatings learned to exploit the noise structure of that specific domain.

### What This Means for the Project

1. **AF evolution is domain-specific.** Evolving on coatings produces AFs that work on coatings but not on ZDT1/DTLZ2. This is expected — the AF is tuned to the domain's noise/landscape structure.
2. **Synthetic benchmarks are not a good generalisation test for real-data AFs.** The noiseless synthetic domains have fundamentally different GP behaviour. A meaningful generalisation test would use another noisy real-data domain.
3. **The noise-sweep is still needed.** To test the floor rule causally, we need to vary the observation noise on ZDT1/DTLZ2 and observe whether the floor rule reverses. This is pending.

---

## 15. Dead Ends and Negative Results (Pre-Evolution Work)

Before the AF evolution work, the project went through several rounds of analysis and hypothesis testing on the broader question of strategy adaptation in SDL campaigns. All of these led to negative results or dead ends:

### 15.1 The Composition Pilot (Within-Batch Joint Composition)

**Hypothesis:** The K mechanism (strategy adaptation) operates through within-batch joint composition — the AF's batch selection implicitly composes exploration and exploitation through the diversity of the batch, not through an explicit adaptive weight.

**Test:** Built a narrow pilot comparing joint composition (selecting a diverse batch that includes both exploitative and exploratory candidates) against independent scoring (each candidate scored independently, top-k selected).

**Result:** NULL on both mAb and coatings. The gate closed. Joint composition does not explain strategy adaptation.

### 15.2 The Generation Gate

**Hypothesis:** The AF's candidate generation step (U-NSGA-III) is where adaptation happens, not the scoring step.

**Test:** Compared different candidate generation strategies with the same scoring AF.

**Result:** Closed negative. The generation step does not drive adaptation.

### 15.3 The Batch-Size Ablation

**Hypothesis:** Batch size modulates the exploration-exploitation trade-off — larger batches allow more exploration, smaller batches force exploitation.

**Test:** Ran campaigns with batch sizes 1, 3, 5, 10 and compared HV trajectories.

**Result:** Closed negative. Batch size does not explain the strategy adaptation effect.

### 15.4 The Noise-Integration Gate

**Hypothesis:** Integrating noise information into the AF (e.g. weighting by posterior variance) is what drives adaptation on noisy domains.

**Test:** Compared AFs that use posterior std vs AFs that don't.

**Result:** Closed negative. The noise integration as implemented did not produce adaptation.

### 15.5 Joint Composition Take 2

**Hypothesis:** A revised version of the composition pilot with better controls.

**Result:** Closed negative again.

### 15.6 DA-COREG Alone

**Hypothesis:** A domain-aware co-regionalisation approach (DA-COREG) could capture cross-objective correlations that standard per-objective GPs miss.

**Result:** Closed negative. DA-COREG alone did not improve over the standard per-objective GP approach.

### 15.7 The mAb Decomposition Finding

The user provided new mAb decomposition data showing that on mAb, **scoring barely matters** — the U-NSGA-III diversity selection (which ensures the candidate pool is diverse) is doing the real work, not the AF's scoring. This is the mAb analogue of mu_sum dominance on coatings: the AF's scoring doesn't change the outcome much because the candidate pool is already diverse enough that any reasonable scoring produces similar batches.

This finding redirected the project toward understanding when AF structure matters (mAb with noise-robust fitness) vs when it doesn't (coatings with mu_sum dominance, mAb with U-NSGA-III diversity dominance).

### 15.8 The LLaMEA-BO Conflict

The user identified that LLaMEA-BO (an existing paper) already uses an LLM to rewrite the optimiser, which undercuts the novelty of "K = LLM rewrites the optimiser." The project pivoted from "LLM rewrites the optimiser" to "LLM evolves the acquisition function" — a narrower, more defensible contribution.

### 15.9 The Long-Horizon Escape Hatch

When the evolution stagnated, one possible response was "maybe the AF needs more experiments to show its advantage." The user rejected this: real SDL campaigns are short-horizon by premise (40 experiments, 6 batches). An AF that only works with 200 experiments is not useful for SDLs. The short-horizon constraint is non-negotiable.

### Summary of Dead Ends

| Hypothesis | Test | Result |
|------------|------|--------|
| Within-batch joint composition | Composition pilot (mAb + coatings) | NULL |
| Candidate generation drives adaptation | Generation gate | Closed negative |
| Batch size modulates trade-off | Batch-size ablation | Closed negative |
| Noise integration in AF | Noise-integration gate | Closed negative |
| Joint composition (revised) | Take 2 | Closed negative |
| DA-COREG cross-objective | DA-COREG alone | Closed negative |
| LLM rewrites the optimiser | Literature check | Undercut by LLaMEA-BO |
| Long-horizon needed | Rejected by premise | Not pursued |

All of these dead ends are valuable: they rule out explanations and narrow the search space. The AF evolution work is the surviving hypothesis — that the LLM can evolve better acquisition functions, and the design rules for when different AFs work can be derived from systematic analysis.

---

## 16. Design Rules Derived

### Rule 1: mu_sum Dominance on Coatings

On the coatings domain, any nonzero uncertainty weight (fixed, progress-decaying, or stagnation-boosted) produces the same candidate selections. 7/8 AFs in Run 4 had identical held-out HV. The sigma term doesn't change rankings. The improvement over EGBO-novelty comes from mu_sum exploitation itself, not the uncertainty bonus.

**Implication:** Coatings is the wrong substrate for AF evolution. The evolution can't optimise what it can't distinguish. The AF structure doesn't matter on this domain — only whether you include sigma at all.

**Caveat:** This is specific to the coatings domain (4D, near-independent objectives, moderate noise). It does not generalise to mAb (where AFs are distinguishable) or synthetic benchmarks (where the uncertainty bonus hurts).

### Rule 2: AF Structure Matters on mAb

On the mAb domain, AFs are NOT rank-equivalent. The held-out validation showed a 188 HV spread between AFs (hint_fixed_ucb=8273 vs gen6_child0=8461). The sigma term DOES change rankings. The significant winners are adaptive UCB AFs. trust_only is consistently non-significant.

**Implication:** mAb is the right substrate for AF evolution. The fitness signal can separate AFs. But the noise floor (CV=49.7%) must be addressed first.

### Rule 3: The Floor Rule (Domain-Specific, Not Yet Causally Tested)

On noisy real-data domains (coatings, mAb), adding any uncertainty bonus improves HV by ~3-6% over pure exploitation. On noiseless synthetic domains (ZDT1, DTLZ2), the uncertainty bonus hurts or has no effect.

**Status:** This is a re-description of observed data, not a testable causal rule. The noise-sweep (varying observation noise on ZDT1/DTLZ2) is needed to test whether noise is the causal driver. Pending.

### Rule 4: The Stagnation Term Is a Liability

AFs with a stagnation novelty boost (gen4_child0, gen1_child1) dropped below significance under seed averaging, while AFs with simple progress-only decay (gen5_child0, gen9_child0) held and strengthened. The stagnation term adds a second source of variability (depends on `stagnant_batches`, which is itself noisy).

**Implication:** Simpler AFs (fewer noisy inputs) are more robust. The evolution should prefer progress-only decay over progress+stagnation. This doesn't require removing the stagnation term from the hint set, but the evolution may naturally prefer the simpler version.

### Rule 5: Seed Noise Masks Real Signal

The hint_fixed_ucb flip (p=0.596 → p=0.030 under 3-seed averaging) proves that single-seed evaluations are dominated by seed noise. Any AF that "wins" or "loses" in a single-seed evaluation could be a seed artefact.

**Implication:** All fitness evaluations must use multi-seed averaging (n_fitness_seeds ≥ 3). The evolution's internal fitness, not just the held-out validation, must use this.

### Rule 6: gamma Must Be Calibrated to the Domain's Margin Magnitude

gamma=0.005 was too aggressive for coatings (margins ~3-5%): 6/8 AFs had negative fitness, evolution stagnated. gamma=0.001 worked (margins ~3.4%): evolution progressed for one generation. The LOC penalty should be small relative to the typical margin difference between AFs.

**Implication:** gamma should be set to ~0.001 × (typical margin magnitude). On mAb (margins ~4-5% but noisy), gamma should be calibrated AFTER the fitness is noise-robust, not before.

### Rule 7: Coatings-Evolved AFs Do Not Generalise

0/8 significant on mAb (cross-domain transfer), 0/8 positive significant on ZDT1/DTLZ2 (synthetic generalisation). The AFs are overfit to the coatings domain.

**Implication:** AF evolution is domain-specific. Generalisation should be tested on other noisy real-data domains, not noiseless synthetic benchmarks.

---

## 17. First-Principles Redesign for Real-World Data

After the noise diagnostic and 3-seed re-validation, the assistant proposed a first-principles redesign of the evolution for real-world data. The user asked for this reasoning. Here is the full proposal:

### 17.1 Oracle: Snap-Query → GP-Oracle

**Current:** The snap-query oracle queries the real dataset at the nearest neighbour. This is deterministic but limited to the 253 (coatings) or 33 (Waibel mAb) real samples.

**Proposed:** Fit a GP to all real data (stochastic — depends on GP-fit seed), then query the GP posterior mean at any point. This gives a continuous oracle with realistic noise structure. Cap the campaign budget at N/2 (half the dataset size) to avoid the campaign exhausting the real data.

**Status:** Untested. The GP-oracle idea is a larger change that could be pursued if the snap-query results are untrustworthy even after the fitness fixes.

### 17.2 Fitness: Mean → Median + AUC

**Current:** `fitness = mean(rel_margins) - gamma * loc`

**Proposed:** `fitness = median(rel_margins) + w * AUC(rel_hv_trajectory) - gamma * loc`

The mean is inflated by outlier campaigns (gen6_child0: mean margin +3.745%, median +0.181% on coatings). The per-campaign margin distribution is wide (std 0.494) and skewed (swings -0.31 to +0.79). The median is more robust to outliers.

Adding AUC of the HV trajectory rewards AFs that improve quickly (useful in short-horizon campaigns) rather than just AFs with high final HV. This is important because two AFs with the same final HV may have very different trajectories — one that improves fast and plateaus is better than one that improves slowly and catches up at the end.

**Status:** The median fitness is validated by the noise diagnostic (the mean is inflated by outliers). The AUC component is proposed but not yet tested.

### 17.3 Transfer: Post-Hoc → In-Fitness

**Current:** Transfer is tested post-hoc (evolve on domain A, validate on domain B).

**Proposed:** Add a transfer penalty to the fitness: `fitness = 0.8 * domain_A_fitness + 0.2 * domain_B_fitness`. This pressures the evolution to find AFs that work on both domains, not just the training domain.

**Status:** Proposed, not yet tested. The 20% weight is a starting point — it should be high enough to matter but not so high that it dominates the primary domain's fitness.

### 17.4 Hints: Add Noise-Aware, Sparsity-Aware, Efficiency-Aware Mechanisms

The current 8-hint set focuses on UCB, novelty, HV improvement, diversity, and trust regions. For real-world data, three additional mechanism families are needed:

- **Noise-aware:** AFs that explicitly account for observation noise (e.g. knowledge-gradient, noisy-EHVI)
- **Sparsity-aware:** AFs that work well with few observations (e.g. trust-region BO, local models)
- **Efficiency-aware:** AFs that consider the cost of experiments (e.g. cost-aware BO)

**Status:** The 3 broken hints need to be repaired first. Then these new mechanism families can be added.

### 17.5 Stopping: Fixed 20 → Stagnation-Based

**Current:** All runs use 20 generations. Runs 1, 2, 3 stagnated at gen 0 — 19 wasted generations. Run 4 stagnated at gen 6 — 14 wasted generations.

**Proposed:** Halt after 8-10 generations of no improvement. This saves compute (the user runs the real LLM locally with a 72B model) and avoids wasting LLM calls on a converged population.

**Status:** Proposed, not yet implemented. The stagnation threshold (8-10 gens) is a starting point.

---

## 18. Real mAb Formulation Data Landscape

Three real mAb formulation datasets were identified for potential use as evolution substrates or transfer test domains:

### 18.1 Radford, Tamasi, Di Mare & Gormley (2026) [20]

- **Paper:** "Active Learning for Biotherapeutic Formulation Development" (Advanced Science, DOI: 10.1002/advs.76551)
- **Data:** Real closed-loop BO on a model antibody (bIgG)
- **Inputs:** 7D continuous (buffer molarity, buffer ID, pH, NaCl, sucrose, arginine, protein concentration)
- **Objectives:** Tm (maximise), diffusivity (maximise), viscosity (minimise)
- **Size:** ~36-48 formulations across 2 DBTL rounds (24 LHS seed + 2×6 per arm)
- **Format:** JSON with per-formulation arrays of concentration/value/std
- **Code:** GitHub: GormleyLab/AL-for-Bioformulation (MIT license)
- **Stack:** Same BoTorch stack (SingleTaskGP, qEI, qEHVI)

This is the most promising real-data anchor: it's a real closed-loop BO campaign with the same objective structure as our mAb oracle, the data and code are publicly available, and it uses the same BoTorch stack. The GP-oracle approach (Section 17.1) could be applied to this data.

### 18.2 Narayanan et al. (2021) [11]

- **Paper:** "Machine Learning for Biologic Formulation Development" (Mol. Pharm., DOI: 10.1021/acs.molpharmaceut.1c00469)
- **Data:** 8 factors, 2 objectives, 3 scFv variants
- **Size:** 25-33 experiments per variant
- **Format:** Data in supplementary information

Smaller than Radford/Gormley but covers a different antibody format (scFv vs full IgG). Useful for transfer testing.

### 18.3 Xin et al. (2024) [4]

- **Paper:** "Formulation Development of Trastuzumab Biosimilar" (Antibody Therapeutics, DOI: 10.1093/abt/tbae028)
- **Data:** 96 formulations, trastuzumab biosimilar
- **Objectives:** HMW (maximise), fragmentation (minimise), viscosity (minimise)
- **Format:** Data in supplementary

The largest dataset (96 formulations) but different objectives (HMW/fragmentation vs Tm/kD/viscosity). Useful for testing whether AFs generalise across objective sets.

### 18.4 Waibel et al. (2025) [30]

- **Paper:** (Mol. Pharm., DOI: 10.1021/acs.molpharmaceut.5c00591)
- **Data:** 33 real formulations, 8 continuous inputs, 3 objectives (Tm max, kD max, RM_Agi min)
- **Each objective has a confidence interval column** — providing real experimental noise
- **Pareto front:** 15/33 = 45.5%
- **Noise-to-signal ratios:** Tm: 0.025 (low), kD: 0.163 (high), RM_Agi: 0.123 (moderate)
- **HV:** ~17,855 (reference point [60, -25, -101] in all-maximise convention)

This dataset was used as the real-data mAb oracle for the cross-domain transfer test and the 3-seed re-validation. The kD objective has the highest noise-to-signal ratio (0.163), meaning the GP's kD predictions are the least reliable — this is the objective where the uncertainty bonus should matter most.

---

## 19. User Corrections Log

Throughout the project, the user made several important corrections to the assistant's analysis. These are logged here because they shaped the conclusions:

### 19.1 Non-Reproducing Mechanistic Claims (Retired)

Two mechanistic claims made by the assistant did not reproduce under scrutiny:
- **Reachable-front count:** A claim about the number of reachable Pareto front points. Retired.
- **CV comparison:** A claim comparing coefficients of variation across domains. Retired.

Both were removed from the analysis. The user's standard: if a claim doesn't reproduce, retire it — don't salvage it.

### 19.2 Dimensionality/Sparsity Causal Claim Downgraded

The assistant claimed that dimensionality and sparsity causally determine AF performance. The user downgraded this to a **hypothesis** — it's a plausible explanation but not yet tested. The causal mechanism (dimensionality → GP sparsity → AF structure matters) is not verified.

### 19.3 Structural Classification Corrected (3/3/1 not 4/2/1)

The assistant initially classified the Run 4 final population as 4 adaptive UCB / 2 fixed UCB / 1 pure exploitation. The user corrected this to 3 progress-decaying UCB / 3 progress+stagnation UCB / 1 fixed UCB / 1 pure exploitation. The user also noted that gen9_child0 was misclassified (it's progress-decaying, not fixed), and that trust_only breaks the adaptive-vs-fixed dichotomy (it's neither — it's pure exploitation with no uncertainty term).

### 19.4 Gamma/LOC Threshold Argument Corrected

The assistant argued that a threshold on gamma would fix the reversal (high-margin AFs losing to low-LOC AFs). The user corrected: the threshold doesn't fix the reversal; the real problem is n=15 training noise. The threshold value should be deferred until after the n_seeds=100 re-run (which became the n=75 run).

### 19.5 Docstring-Fidelity Check Corrected

The assistant proposed a docstring-vs-formula fidelity checker. The user corrected: the contradiction was between **inline comments**, not between the docstring and the formula. A general semantic verifier is a separate follow-up, not something to bundle into the current work.

### 19.6 Margin Metric Is Relative, Not Raw HV

The assistant initially treated the margin as raw HV, leading to the conclusion that gamma=0.005 was "inert" (the penalty was tiny compared to raw HV values of ~400). The user corrected: the margin is a **fraction** (relative HV improvement), so gamma=0.005 is NOT inert — it's 0.5% per LOC line, which is significant when margins are only 3-5%.

### 19.7 Floor Rule Is a Re-Description, Not Yet Testable

The assistant presented the floor rule as a testable design rule. The user corrected: it's a re-description of observed data (noiseless domains don't benefit from uncertainty bonuses), not a testable causal rule. The noise-sweep is needed to make it testable.

### 19.8 mAb Inversion Compounded by Noise Floor

The assistant noted that mAb results were inverted (worse AFs winning). The user corrected: the inversion is compounded by the CV=50% noise floor. The fitness noise-robustness must be fixed BEFORE gamma recalibration — you can't calibrate gamma on a fitness signal that's dominated by noise.

### 19.9 hint_fixed_ucb "Overfits" Narrative Reracted

The assistant initially said hint_fixed_ucb overfits (it was non-significant on held-out with n=5 campaigns). The user retracted this: it was a false negative at n=5, not genuine overfitting. With more held-out campaigns, hint_fixed_ucb was significant.

---

## 20. Current Plan and Next Steps

### The Current Plan (from PLAN.md)

The plan is to evolve on mAb with noise-robust fitness, validated by the 3-seed re-validation:

**Step 1: Implement fitness fixes**
- 1a: Multi-seed averaging (n_fitness_seeds=3) — VALIDATED by the 3-seed re-validation
- 1b: Median fitness (median margin instead of mean margin) — needed for the 71% domain noise

**Step 2: Repair the 3 broken hints**
- local_penalization: fix the 3 bugs (empty scores list, inverted penalty, float index)
- pareto_membership: fix the dominance direction (DOMINATED BY, not DOMINATES)
- noisy_front_hvi: fix the repetition-loop pathology (add "write code immediately, minimal comments")

**Step 3: Evolve on mAb with noise-robust fitness**
- Fitness: median margin, n_fitness_seeds=3
- Stopping: stagnation-based (halt after 8-10 no-improvement generations)
- Hint set: 8-hint set from af_interface_v2.py (with Step 2 repairs)
- gamma: start at 0.001
- Substrate: mAb (excipient oracle) — AFs are distinguishable here

**Step 4: Mandatory 3-seed held-out re-validation**
- Run the final population through validate_population_2b with n_fitness_seeds=3 on 20-25 held-out mAb campaigns
- Only AFs significant (p<0.05) on held-out are candidates for deployment

**Step 5 (deferred): Real-data anchor + synthetic generalisation**
- Radford/Gormley real-data anchor: GP-oracle from their JSON, test transfer of mAb-evolved AFs
- Synthetic generalisation: only if each synthetic dataset is sized to clear the statistical bar

### What the Sandbox Builds vs What the User Runs

- **Sandbox builds (Steps 1-2):** Modified evaluate_af_2b with multi-seed averaging + median fitness; fixed af_interface_v2.py (3 hint repairs)
- **User runs locally (Steps 3-4):** mAb evolution with noise-robust fitness; 3-seed held-out re-validation
- **Deferred (Step 5):** Radford/Gormley oracle + transfer test; synthetic benchmark suite

### Unresolved Items

1. **The noise-sweep on synthetic benchmarks** — varying observation noise on ZDT1/DTLZ2 to test the floor rule causally. Still pending.
2. **The 3 broken hint repairs** — local_penalization, noisy_front_hvi, pareto_membership. Proposed fixes exist but are not yet implemented.
3. **The narrow direction-label checker** — a 10-line static check for comment-formula mismatch. Separate follow-up.
4. **Adding IGD to the project's coatings reporting** — the Aqeeli et al. paper reports IGD, not just HV. Currently not implemented.
5. **The mAb/coatings exploitation-edge mechanism** — what makes the exploitation edge work on mAb but not coatings. Not yet identified.
6. **Overall paper structure** — not yet decided.
7. **The remaining 3 synthetic benchmarks** (ZDT3, ZDT4, CTP2) + noise sweep — pending.
8. **NOSTRA trust-region adaptation and testing** — the NOSTRA-inspired pareto_membership hint drifted; the trust-region concept is untested.
9. **Radford/Gormley real-data anchor** — the GP-oracle + transfer test. Deferred until Steps 1-4 produce a confirmed mAb-evolved AF.
10. **Whether to re-run mAb evolution with different parameters** — depends on Step 3 results.

### Priority Order

1. Write the comprehensive project log document (this document — DONE)
2. Implement fitness fixes (multi-seed averaging + median) in evaluate_af_2b
3. Repair the 3 broken hints in af_interface_v2.py
4. Re-run mAb held-out validation with fixed fitness
5. Evolve on mAb with noise-robust fitness (if step 4 confirms signal)
6. Radford/Gormley real-data anchor + transfer test
7. Synthetic generalization (only if justified)

---

## 21. Post-Hoc Follow-Up: DA-COREG Without qLogNEHVI (2026-08-01)

### Motivation

§8's Part 8 DA-COREG result (−3.8%, 0/20 wins, p≈1.9×10⁻⁶ on DTLZ2) left one
mechanism untested: `run_da_coreg_pilot.py`'s ablation kept qLogNEHVI-optimized
`qbo_x` in the candidate pool and used qLogNEHVI itself for scoring in **every**
cell (deliberately, to hold candidate distribution fixed within that 2×2), so it
could never separate "DA-COREG's posterior is bad" from "DA-COREG's posterior is
fine but qLogNEHVI's correlated joint MC batch scoring specifically punishes a
MultiTaskGP posterior" — the leading, unproven hypothesis from Part 8's
mechanism-elimination process.

### Setup

`full_replay.strategy_unsga3_pool_af` gained a `use_da_coreg` flag, swapping its
independent `ModelListGP` fit for `compose_strategies._fit_model`'s DA-COREG
branch, with everything else (UNSGA3-only candidate generation, `trust_only`
`score_pool` scoring, top-k `select_batch` — **no `optimize_acqf`/qLogNEHVI call
anywhere in the pipeline**) held identical. New script
`run_da_coreg_no_qnehvi_pilot.py` compares `unsga3_pool_af_indep` vs
`unsga3_pool_af_da_coreg` on the same DTLZ2 positive control, same harness
(3 replicates × 20 campaigns, Wilcoxon per replicate) as the original test.

### Result

| | qLogNEHVI-in-the-loop (original, §8 Part 8) | qLogNEHVI removed (this follow-up) |
|---|---|---|
| Mean margin | −3.8% | −1.6% (std 0.7 across replicates) |
| Win rate | 0/20 (every replicate) | 10/20, 5/20, 8/20 |
| p per replicate | 1.9×10⁻⁶ | 0.812, 0.040, 0.154 |
| Replicates significant | 3/3 (all catastrophic) | 1/3 (modest, −1.9%) |
| Replicates directionally positive | 0/3 | 0/3 |

### Interpretation

The catastrophic, always-loses signature (0/20 wins, p≈10⁻⁶, every replicate)
does **not** reproduce once qLogNEHVI is removed from the pipeline — this
supports the "qLogNEHVI's joint MC batch scoring is what punishes DA-COREG"
hypothesis over "DA-COREG's posterior is bad on its own." But the result isn't
clean: DA-COREG is still directionally negative in 3/3 replicates even without
qLogNEHVI, and the between-replicate instability (p swinging from 0.81 to 0.04
across nominally-identical setups) matches the seed-noise signature documented
in §12/§13 — the same class of noise (GP-fit seed + UNSGA3 seed, uncancelled by
seed-pairing because the two conditions' trajectories decorrelate after batch 1)
is almost certainly present here too, since this pilot never decoupled or
averaged over it.

**Status: real qualitative shift, not yet resolved to "ties" vs. "small real
cost."** `run_da_coreg_no_qnehvi_pilot.py` gained an `--n_fitness_seeds` flag
(averaging each campaign's final_hv over N seeds before computing wins/diffs,
same rationale as §12/§13) to resolve this; not yet run at n_fitness_seeds>1 as
of this entry (cost multiplies linearly — a single n_fitness_seeds=3 run at the
existing 3×20-campaign scale would run roughly 3× the ~550s/replicate observed
here, so ~1600s/replicate, ~80 minutes for 3 replicates).

### Implication for the mechanistic hypothesis in Part 8

This is the first piece of direct evidence (rather than ruled-out alternatives)
for the "qLogNEHVI compresses acquisition-value discrimination on a MultiTaskGP
posterior" hypothesis. It should still be treated as provisional until the
seed-noise question above is resolved — but note this reframes Rule of thumb
implications from §16: DA-COREG may be viable specifically in evolved-AF /
score_pool-style pipelines (which never call qLogNEHVI) even though it remains
inadvisable to pair with the qLogNEHVI-based baseline/ablation-cell pipelines
tested everywhere else in this project.

---

## 22. Tunable Synthetic Domain: Front-Range Growth, Not Shrinkage (2026-08-06)

### Motivation

`run_tunable_domain_generalization.py`'s paired-Wilcoxon comparison of
`gen6_child0_tuned` (front-range-normalised UCB, β=15 on σ/front_range) vs.
`hint_fixed_ucb` (raw UCB, β=2 on raw σ) on `TunableSyntheticMOOracle` — a
domain purpose-built (per its module docstring) to give front-range
normalisation something to bite on — came back non-significant (12/20,
p=0.15) despite both individually beating `trust_only`. `track_front_range.py`
was written to test one specific hypothesis for the wash: front-range
normalisation's claimed edge over a fixed-β UCB is that it's *implicitly
self-annealing* — σ/front_range should grow in relative weight as the
Pareto front narrows over the campaign. If `front_range` barely moves, the
two AFs are nearly the same function up to a constant, which would exactly
explain a null result.

### Setup

Two passes, deliberately run both ways to separate "is this a real domain
mechanism" from "is this a sklearn-vs-botorch GP artifact":

1. A lightweight, dependency-free stand-in
   (`generate_bo_diagnostics.py`, new this entry) — same oracle, same two AF
   formulas, but sklearn `GaussianProcessRegressor` GPs and greedy top-k
   batch selection instead of BoTorch/qLogNEHVI (torch unavailable in that
   environment). 12 campaigns, `plateau_sharpness=5.0`, `noise_level=0.08`,
   `noise_mode="proportional"`, `scale2=3.0`, `budget=40`, `n_init=10`,
   `batch_size=5` — the same operating point `sweep_tunable_domain.py`
   selected.
2. The real harness: `track_front_range.py` unmodified, same params, 12
   campaigns, run against the actual `full_replay.strategy_evolved_af` /
   BoTorch GP campaign loop.

Both log `pareto_front_range` per objective, per batch — the same
`context["pareto_front_range"]` field `score_pool` divides by.

### Result

Per-batch mean `front_range`, batch 0 → batch 6 (init-only → final):

| | f1, sklearn stand-in | f1, real BoTorch harness | f2, sklearn stand-in | f2, real BoTorch harness |
|---|---|---|---|---|
| `hint_fixed_ucb` | 0.731 → 0.978 (+34%) | 0.755 → 0.974 (+29.0%) | 2.192 → 3.160 (+44%) | 2.869 → 3.344 (+16.5%) |
| `gen6_child0_tuned` | 0.731 → 1.002 (+37%) | 0.755 → 0.952 (+26.1%) | 2.192 → 3.517 (+60%) | 2.869 → 3.616 (+26.0%) |

`front_range` **grows monotonically for both conditions, on both harnesses** —
it never shrinks over the campaign. The growth rate and absolute values
differ somewhat between the sklearn stand-in and the real harness (expected,
given the cruder GP fit and greedy — not joint — batch selection in the
stand-in), but the direction and the "both conditions nearly identical"
pattern replicate exactly.

Hypervolume (sklearn stand-in only — the real harness's HV comes from
`run_tunable_domain_generalization.py`'s existing non-significant result):
raw UCB 12.76→13.70, front-range-norm 12.76→13.84 across the same 6 batches —
consistent with the two AFs tracking each other closely throughout, not just
at the final HV comparison.

A rendered comparison (HV convergence, front-range trace with both harnesses
overlaid, derived effective-β trace, campaign-0 objective-space fill, and the
`sweep_tunable_domain.py` dominance_ratio grid) is at
`generate_bo_diagnostics.py`'s output artifact — see that script's docstring
for regeneration instructions.

### Interpretation

The self-annealing premise front-range normalisation needs — "the front
narrows as the campaign converges, increasing σ/front_range's relative
weight" — **does not hold on this domain at `n_init=10`**. The mechanism runs
backward: with only 10 initial points, the *observed* non-dominated front
starts artificially small (a sparse sample rarely contains the true extremes
of a 500-point pool), then widens monotonically as the campaign discovers
more of the actual Pareto set. So `effective_beta = 15/front_range` *falls*
over the campaign instead of rising — the opposite of the intended annealing
direction — for both conditions almost identically, which is consistent with
why the two AFs are statistically indistinguishable on this domain despite
both being real, if modest, improvements over `trust_only`.

This is a **harness-independent** finding (confirmed on both the sklearn
stand-in and the real BoTorch/qLogNEHVI harness) and does not depend on
`sweep_tunable_domain.py`'s noise/dominance_ratio tuning — it's a property of
how the observed front's extent evolves with campaign progress on this
domain, not a symptom of the wrong `(plateau_sharpness, noise_level)` cell.

### Open question

Whether `front_range` eventually turns over and shrinks with a larger
`n_init` (a bigger initial sample would capture more of the true front's
extremes up front, leaving less room to grow) or longer budget, or whether
unbounded growth is structural to this ZDT1-family oracle's 500-point finite
pool regardless of budget, is untested as of this entry — a sweep over
`n_init` (holding `budget` fixed) would resolve it directly.

---

## 23. Confirmatory Robustness Spec for Front-Range Normalisation (2026-08-07)

### Motivation

§22 explained *why* `gen6_child0_tuned` (β=15) vs. `hint_fixed_ucb` (β=2) came
back non-significant on the tunable domain, but a paired-Wilcoxon head-to-head
on a single hand-picked β and a single oracle seed was never more than
exploratory. Before treating the tunable domain's earlier positive signals
(15/20 wins, p=0.008 at budget=20; §21-adjacent sweep results) as evidence
for the thesis, the claim needed a properly powered, pre-specified
confirmatory test — designed so a null result would be as informative as a
positive one, and so a positive result couldn't be read as p-hunting over an
unprincipled hyperparameter.

### Spec design (wayfinder map, `.scratch/front-range-robustness-spec/`)

Planned via the wayfinder skill as seven resolved tickets, each a locked-down
design decision (full detail in each ticket file; map at
`.scratch/front-range-robustness-spec/map.md`):

1. **Primary endpoint** — `log(HV_true − HV_observed)` AUC over batches
   (matches BoTorch's own MOBO benchmarking convention), vs. the true-optimum
   HV computed from the oracle's noiseless pool. Secondary: batches-to-90%-
   of-optimum. Tertiary: final HV at budget=40 (carries §21's existing null).
2. **Domain-seed replication** — 8 independent domain seeds
   (`{42..49}`, same `plateau_sharpness/noise_level/noise_mode/scale2` fixed
   across all of them), random-intercept `statsmodels.MixedLM`
   (`auc ~ condition + (1|domain_seed)`) as the primary test instead of
   per-campaign Wilcoxon — collapses the pseudoreplication problem of
   treating 20 campaigns on one oracle draw as 20 independent trials.
3. **Beta sweep grid** — `gen6_child0_tuned` swept over
   β∈{2, 5, 10, 15, 25}; `hint_fixed_ucb` fixed at β=2 (not swept).
   Robustness = same-sign favouring gen6_child0 in ≥4/5 betas **and**
   Benjamini-Hochberg-significant in ≥3/5.
4. **Mechanism isolation** — compare against `phase_decaying_ucb` (existing
   `SEED_PROGRAMS` entry, explicit hand-tuned exploration decay) via TOST
   equivalence testing (δ = 20% of the primary effect size — the FDA/EMA
   bioequivalence and Lakens Cohen's-d=0.2 conventions both land near this),
   not ordinary non-significance.
5. **Statistical correction** — Benjamini-Hochberg per-beta across batches
   (not Bonferroni over the whole grid), domain-seed-level cluster bootstrap
   for CIs (never a flat pooled bootstrap, which would pseudoreplicate the
   same way per-campaign Wilcoxon did).
6. **Real-domain tie-back — ruled out of scope.** Coatings (253 real
   samples) is structurally blocked by `mu_sum` dominance at any budget
   (7/8 AFs rank-equivalent in the original thesis diagnosis); the "mAb"
   oracle (`DiscreteMOExcipientOracle`) is synthetic, not real data, and
   carries a CV≈49.7% noise floor on top of an already budget-fragile effect.
7. **Pre-registration boundary** — the exploratory pilot (seed=42, β=15.0
   only) is hypothesis-generating/parameter-calibrating only; its own
   p-values are never cited as confirmatory, shown explicitly in write-ups
   as a labelled non-confirmatory preamble.

Executed by `run_confirmatory_spec.py` (8 seeds × 20 campaigns × budget=40 ×
7 conditions = 7,840 batch-rows; ~4.5hr wall time).

### Result: robustness criterion NOT MET

| β | effect (is_gen6 on AUC) | p |
|---|---|---|
| 2 | **+0.316** | **0.003** |
| 5 | +0.159 | 0.11 |
| 10 | −0.016 | 0.87 |
| 15 | −0.047 | 0.63 |
| 25 | −0.071 | 0.47 |

Same-sign favouring gen6_child0 in 3/5 betas (need ≥4/5); BH-significant in
1/5 (need ≥3/5). **Criterion not met.** The pilot's calibrated β=15 does
*not* replicate across 8 independent domain seeds — its bootstrap CI vs.
`hint_fixed_ucb` spans [−0.15, +0.12], straddling zero. The TOST check
against `phase_decaying_ucb` at β=15 also failed to show equivalence
(p=0.9998 at δ=0.0094 — nowhere close). Only β=2 — numerically identical to
`hint_fixed_ucb`'s own fixed β — shows a significant, sizeable effect.

### Interpretation and follow-up

β=15 (the pilot's hand-picked, dominance_ratio-motivated value) does not
generalise; treating it as validated would have been exactly the p-hunting
the confirmatory design exists to catch. That β=2 alone is significant is
itself informative but ambiguous two ways: (a) front-range normalisation's
real benefit doesn't require — and is in fact hurt by — scaling β up to
compensate for a domain's `dominance_ratio`, since `sigma_norm` is already
doing that scale-correction implicitly; or (b) the β=2 result is itself a
seed-lucky false positive that a further replication (a tighter grid) would
wash out.

Two follow-ups launched to distinguish these, both committed on
`add-tunable-synthetic-domain` and ready to run at the same 8×20×40 scale:

- **`run_confirmatory_spec.py --betas 1,2,3 --calibrated_beta 2`** — a
  narrow low-β sweep (script generalised with `--betas`/`--calibrated_beta`
  CLI overrides, robustness thresholds auto-scaled to grid size) to check
  whether the β=2 effect is a plateau near baseline-β or a narrow, possibly
  noisy peak at exactly β=2.
- **`run_gpucb_schedule.py`** — removes β as a tuned parameter entirely.
  Weights `sigma_norm` by the theoretically motivated GP-UCB confidence
  schedule (Srinivas et al. 2010, finite-domain form:
  `beta_t = 2*log(|pool|*t²*π²/(6δ))`, δ=0.1 fixed by convention, not fit
  to this domain) instead of any fixed/swept multiplier. If this beats
  `hint_fixed_ucb` with no tuning at all, that is a materially stronger and
  cleaner claim than any beta-grid result could be — "front-range
  normalisation + a standard confidence-bound schedule, no domain-specific
  calibration, robustly outperforms fixed-β UCB" — and would be the
  headline result to report rather than any single swept β.

Neither has been run to completion as of this entry (§23); both are queued.

---

## Appendix A: Key Citations

| Ref | Paper | Relevance |
|-----|-------|-----------|
| [1] | NOSTRA (Ghasemzadeh et al., 2025, arXiv 2508.16476) | Trust-region BO, inspired the pareto_membership hint |
| [4] | Xin et al. 2024 (Antibody Therapeutics, DOI: 10.1093/abt/tbae028) | 96-formulation trastuzumab biosimilar dataset |
| [5/17] | qNEHVI (Daulton et al., 2021, arXiv 2105.08195) | Batch EHVI acquisition, the baseline AF |
| [6] | Adaptive replication (Binois & Larson, 2025, arXiv 2504.20527) | Noise-aware replication strategy |
| [7] | MORBO (Daulton et al., 2021, arXiv 2109.10964) | Multi-objective trust-region BO |
| [11] | Narayanan et al. 2021 (Mol. Pharm., DOI: 10.1021/acs.molpharmaceut.1c00469) | 8-factor, 2-objective scFv formulation dataset |
| [12] | PDBO (Ahmadianshalchi et al., 2024, AAAI, DOI: 10.1609/aaai.v38i10.28951) | Pareto-dominance BO |
| [13] | DPP-BBO (Nava/Mutný/Krause, 2021, arXiv 2110.11665) | DPP-based diversity for batch BO, inspired the dpp_diversity hint |
| [14] | DPP-Batch BO (Kathuria et al., 2016, arXiv 1611.04088) | DPP for batch selection in BO |
| [19] | EGBO (Low et al., 2024, npj Computational Materials, DOI: 10.1038/s41524-024-01274-x) | EGBO with novelty selection, the baseline |
| [20] | Radford et al. 2026 (Advanced Science, DOI: 10.1002/advs.76551) | Real mAb formulation BO dataset, GitHub: GormleyLab/AL-for-Bioformulation |
| [30] | Waibel et al. 2025 (Mol. Pharm., DOI: 10.1021/acs.molpharmaceut.5c00591) | 33-sample mAb dataset used as the excipient oracle |

## Appendix B: File Inventory

### Evolution Run Data (User Uploads)

| File | Run | Contents |
|------|-----|----------|
| `run_v2_coatings_real/final_population.json` | Run 1 | 8 AFs, gamma=0.005, n=15, coatings |
| `run_v2_coatings_real/history.json` | Run 1 | 20-gen trajectory, 40 LLM calls |
| `final_population.json` | Run 2 | 8 AFs, gamma=0.005, n=75, coatings |
| `run_v2_mAb_real_100/final_population.json` | Run 3 | 8 AFs, gamma=0.005, n=75, mAb |
| `run_v2_mAb_real_100/history.json` | Run 3 | 20-gen trajectory, 40 LLM calls |
| `run_v2_coatings_gamma001_v2/final_population.json` | Run 4 | 8 AFs, gamma=0.001, n=75, 8-hint, coatings |
| `run_v2_coatings_gamma001_v2/history.json` | Run 4 | 20-gen trajectory, progressed at gen 6 |
| `run_v2_coatings_gamma001_v2/best_af.py` | Run 4 | gen6_child0 (the winner) |
| `run_v2_coatings_gamma001_v2/af_code_logs/` | Run 4 | 49 LLM call logs (call_00000 through call_00048) |

### Validation Data

| File | Contents |
|------|----------|
| `synthetic_generalization_zdt1_results.json` | ZDT1 results (coatings AFs, n=20) |
| `synthetic_generalization_dtlz2_results.json` | DTLZ2 results (coatings AFs, n=20) |
| `waibel_mAb_formulation_dataset_CORRECTED.csv` | 33-sample real mAb dataset |

### Code

| File | Contents |
|------|----------|
| `/mnt/results/af_interface_v2.py` | 8-hint set (v2.1) |
| `/mnt/results/ls_na_egbo_code/` | 16 .py files + README.md (the BO campaign infrastructure) |
| `/mnt/results/execution_trace/PLAN.md` | Current plan (mAb evolution with noise-robust fitness) |
| `/mnt/results/execution_trace/worker-0.ipynb` | Notebook with all analysis cells (105 cells) |

### Notebook Cell Map (worker-0.ipynb, 105 cells)

| Cells | Content |
|-------|---------|
| 0-49 | Early benchmark analysis (Phase 1/2, sensitivity, power analysis, evolution simulation) |
| 50-59 | 3-replicate coatings meta-analysis, corrected findings |
| 60-66 | K vs LLaMEA-BO, gamma recalibration analysis |
| 67-76 | Run 1 analysis, structural patterns, gamma argument |
| 77-85 | Aqeeli et al. coatings analysis, HV/IGD split |
| 86-94 | Run 2 (n=100 coatings), 5 significant winners, cross-domain transfer |
| 95-99 | mAb evolution + synthetic generalization + definitive synthesis |
| 100-101 | Hint feasibility analysis, hint merge analysis |
| 102 | Run 4 analysis (structural classification, held-out, bugs, trajectory) |
| 103 | Noise diagnostic + mAb held-out validation |
| 104 | 3-seed mAb re-validation comparison |

---

*End of document.*
