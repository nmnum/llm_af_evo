# Plan: LLM-Seeded Novelty-Aware EGBO for Multi-Objective Formulation Optimization

## Summary

Design and benchmark an LLM-augmented optimization architecture that beats EGBO on sample efficiency for multi-objective biologic formulation optimization (max Tm, max kD, min viscosity). The architecture uses a local LLM via Ollama (**qwen3:32b**, switched from qwen2.5:72b-instruct — see Model selection below) as a **one-shot warm-start generator** before experiments begin, then hands off to **novelty-aware EGBO** (qLogNEHVI + U-NSGA-III + novelty-weighted batch selection) for the remainder of the campaign. This directly implements the "prior-knowledge routing" recommendation from the existing mechanistic diagnosis while incorporating the current EGBO state-of-the-art (novelty-aware selection from Aqeeli et al. 2026).

### Model selection: qwen3:32b (switched from qwen2.5:72b-instruct)

Measured directly on this project's hardware (NVIDIA GB10 / DGX-Spark-class unified-memory
system) using the real warm-start prompt (25 candidates, full JSON schema):

| Model | Latency (1 call) | Parsed / 25 | Notes |
|---|---|---|---|
| `qwen2.5:72b-instruct` | 554.6s | 24 | works, but ~23hrs for full Phase 2 LLM calls alone |
| **`qwen3:32b`** | **327.4s** | **25** | **selected** — 1.7x faster, no workarounds needed |
| `qwen3.6:35b-a3b` (MoE) | n/a | 0 | thinking mode un-disableable (known Ollama bug: API `think:false` and prompt-level `/no_think` both fail), all output consumed by reasoning trace |

`qwen3:32b`'s benchmark-reported quality is close to `qwen2.5:72b-instruct`'s per Qwen's own
published efficiency claims, so this is treated as a speed win without an assumed quality
cost — not yet independently verified on formulation-quality/diversity beyond the single
smoke-test comparison above.

The key hypothesis: the LLM's domain knowledge (excipient roles, mechanism-pathway mapping) produces a more informative initial design than random LHS, which improves GP lengthscale estimates at small N, which compounds with novelty-aware EGBO's diversity preservation to achieve higher hypervolume in fewer experiments.

## Architecture Decision: LLM-Seeded Novelty-Aware EGBO (LS-NA-EGBO)

### Why this architecture (evidence-based)

Three findings drive this choice:

1. **"When Do LLMs Improve BO?" (NeurIPS 2025)**: Off-the-shelf reasoning LLMs **fail on SMILES** (poor representation handling) but **excel on protein motifs** (simple, LLM-comprehensible representations, less-constrained search spaces). GPT-5 found 11 top-0.5% sequences in 60 evaluations. The formulation problem — named excipients with roles, not SMILES strings — is exactly the LLM-comprehensible regime where reasoning models shine.

2. **Cissé et al. (BORA, Digital Discovery 2026)**: LLM/BO hybrids outperform BO alone, but the advantage is **concentrated in the warm-start phase** (first ~50 experiments). After that, BO catches up. For N<50 campaigns (the user's regime), the LLM's warm-start advantage IS the main advantage. Calling the LLM every batch adds cost without proportional benefit.

3. **User's mechanistic diagnosis**: Reactive routing fails at N<100 for three independent structural reasons (LLM capability, signal non-predictivity, budget trade-off). The recommended fix is **prior-knowledge routing**: LLM decides before experiments start, not reactively during.

### Why NOT the existing strategy_mo_llm or the user's original "LLM generates 20-30 → acquisition selects" proposal

- **Existing strategy_mo_llm**: Calls the LLM every batch with trust-weighted mixing. The trust diagnostic is unreliable at small N (collapses to ~0.85 for every objective, as noted in the code). The LLM is called ~8 times per seed, adding ~4 minutes per seed of latency, for a mixing weight that's largely cosmetic.
- **LLM-as-candidate-generator (user's original proposal; originally mislabeled "LABO-style" — it does not implement the actual LABO paper's multi-fidelity Kennedy-O'Hagan gating mechanism, see `strategy_llm_candidate_gen.py` docstring)**: LLM generates 20-30 candidates every batch, GP scores them. At small N, the GP's acquisition function is unreliable, so "GP scores LLM candidates" is not much better than "LLM picks directly." The Cissé et al. data shows this approach's advantage is mainly in the warm-start phase anyway.
- **LS-NA-EGBO**: LLM is called **once** per seed (to generate the initial batch), then EGBO runs autonomously. This is faster, cleaner, and directly addresses the mechanistic diagnosis's recommendation.

### Architecture specification

```
Stage 1: LLM Warm-Start (experiments 1 to N_init)
  Input to LLM:
    - Excipient catalogue (name, type, range, step_size, roles)
    - Protein degradation pathway (aggregation-prone vs oxidation-prone)
    - Three objectives (max Tm, max kD, min viscosity) and their mechanisms
    - Instruction: generate N_propose diverse formulations spanning the
      mechanism space — some targeting each objective, some trade-offs,
      some "exploration" formulations (e.g., wrong-pathway excipient to
      confirm irrelevance)
  Output: N_propose named formulations (structured JSON)
  Post-processing:
    - Snap concentrations to valid steps
    - Diversity filter (Kennard-Stone or max-min distance in 16D space)
    - Select N_init most diverse formulations
  
Stage 2: Novelty-Aware EGBO (experiments N_init+1 to budget)
  - Per-objective GPs (Matern 5/2, one per objective), warm-started on
    LLM-selected initial points
  - qLogNEHVI acquisition (BoTorch) + U-NSGA-III evolutionary candidates
    (pymoo), merged into a single candidate pool
  - Novelty-aware batch selection: sequential selection with
    score = w_acq * acquisition_merit + w_nov * novelty_distance
    (w_acq=0.7, w_nov=0.3, per Aqeeli et al. 2026)
  - Batch size = 5 (matching Waibel et al.)
```

## Experimental Conditions

All conditions share the same oracle, same noise levels, same budget, and same seeds. Conditions differ only in initialization and/or acquisition strategy.

| Condition | Init | Acquisition | Purpose |
|-----------|------|-------------|---------|
| `mo_random` | Random LHS | Random | Floor baseline (exists) |
| `mo_egbo` | Random LHS | Per-objective GP + Pareto | Lightweight EGBO (exists) |
| `mo_egbo_real` | Random LHS | qLogNEHVI + U-NSGA-III | Real EGBO (exists) |
| `mo_egbo_novelty` | Random LHS | qLogNEHVI + U-NSGA-III + novelty selection | Current SOTA (NEW) |
| `mo_ls_na_egbo` | LLM warm-start | qLogNEHVI + U-NSGA-III + novelty selection | **Recommended architecture** (NEW) |
| `mo_llm_candidate_gen` | Random LHS | LLM generates 20-30 candidates every batch → qLogNEHVI scores | User's original proposal (NEW); NOT literature LABO |
| `mo_llm_existing` | Random LHS | Trust-weighted LLM/GP mixing every batch | Existing strategy_mo_llm (exists) |

Key comparisons:
- `mo_egbo_real` vs `mo_egbo_novelty`: Does novelty-aware selection help? (Aqeeli et al. replication)
- `mo_egbo_novelty` vs `mo_ls_na_egbo`: **The key comparison** — does LLM warm-start beat random init when both use the same SOTA acquisition?
- `mo_ls_na_egbo` vs `mo_llm_candidate_gen`: Warm-start architecture vs throughout-campaign architecture
- `mo_ls_na_egbo` vs `mo_llm_existing`: Recommended architecture vs existing implementation

## Validation Phases

### Phase 1: Reduced formulation space (fast iteration)
- Search space: 1 AA × 1 sugar × 1 surfactant ± EDTA = 4 categorical choices × continuous ranges
- Oracle: `MultiObjectiveExcipientOracle` with reduced excipient lists
- Budget: 20, 30, 40 experiments (sample efficiency curves)
- N_init: 5 (LLM generates 15, select 5 most diverse)
- Batch size: 5
- Seeds: 20 per condition
- Protein profiles: mAb_aggregation, mAb_oxidation (tests mechanism-specific warm-start)
- Prior levels: L1 (correct mechanism), blank (no domain knowledge)
- Purpose: validate architecture, tune novelty weight, test LLM prompt quality

### Phase 2: Full 14-excipient space (real problem)
- Search space: 5 AAs × 4 sugars × 3 surfactants × EDTA = 120 categorical combos × continuous ranges (16D encoded)
- Oracle: existing `MultiObjectiveExcipientOracle` with full catalogue
- Budget: 30, 40, 50 experiments
- N_init: 10 (LLM generates 25-30, select 10 most diverse)
- Batch size: 5
- Seeds: 15 per condition
- Protein profiles: mAb_aggregation, mAb_oxidation
- Prior levels: L1 (correct), blank, wrong (crossed mechanism — tests robustness)
- Purpose: demonstrate the architecture scales to the real problem

### Phase 3: Cross-domain generalisability (Ada coatings)
- Dataset: Ada coatings data (pareto_20210112, 4D continuous)
- Architecture: same LS-NA-EGBO, different LLM prompt (coatings-specific prior knowledge from existing PRIOR_KNOWLEDGE dict)
- Budget: 50% of dataset
- Seeds: 20 per condition
- Purpose: show the architecture generalises without code changes, only prompt changes

## Implementation Details

### New code to write

1. **`llm_warmstart.py`** (~150 lines): LLM warm-start module
   - `llm_generate_initial_formulations(prompt, n_propose, model)` → list of named formulations
   - `diversity_select(candidates, n_select, bounds)` → Kennard-Stone selection in 16D space
   - Prompt template with excipient catalogue, roles, protein pathway, objective mechanisms
   - Structured JSON output parsing (reuse existing parsing from `excipient_campaign_mo.py`)
   - Mock mode for testing (deterministic diverse sampling)

2. **`novelty_selection.py`** (~80 lines): Novelty-aware batch selection
   - `novelty_aware_select(candidates, acquisition_scores, n_select, w_acq=0.7, w_nov=0.3)`
   - Sequential selection: pick highest combined score, recompute novelty distances, repeat
   - Novelty = min Euclidean distance to already-selected points in normalised input space

3. **`strategy_ls_na_egbo.py`** (~120 lines): The recommended strategy
   - Stage 1: call `llm_warmstart` for initial batch
   - Stage 2: call `strategy_mo_egbo_real` with novelty-aware selection patched in
   - Integrates with existing `run_mo_campaign` infrastructure

4. **`strategy_llm_candidate_gen.py`** (~100 lines): LLM-as-candidate-generator strategy for comparison (originally mislabeled "LABO-style"; does not implement the literature LABO gating mechanism)
   - Every batch: LLM generates 20-30 candidates → merge with EGBO candidates → qLogNEHVI scores all → select top batch_size
   - Reuses existing `llm_propose_mo_formulations` from `excipient_campaign_mo.py`

5. **`run_benchmark.py`** (~200 lines): Unified benchmark runner
   - Runs all 7 conditions across all phases
   - Collects: HV trajectory, Pareto front size, IGD (oracle ground truth), experiments-to-X%-HV
   - Statistical tests: Wilcoxon with Holm correction across conditions
   - Outputs: CSV summary, HV trajectory plots, Pareto front visualisations

### Code to modify

6. **`excipient_campaign_mo.py`**: Add `mo_egbo_novelty` and `mo_ls_na_egbo` to STRATEGIES dict; add novelty-aware selection to `strategy_mo_egbo_real`

### LLM prompt design (the critical component)

The warm-start prompt has three layers (matching the generalisability analysis):

**Layer 1 — Reasoning protocol (domain-agnostic, generalisable):**
```
You are designing initial experiments for a multi-objective optimization
campaign. You will track THREE objectives simultaneously. There is no
single "best" formulation — you are looking for good TRADE-OFFS across
objectives (a Pareto front).

Propose {n_propose} diverse formulations that:
  1. Target each objective individually (some maximising Tm, some
     maximising kD, some minimising viscosity)
  2. Include trade-off formulations (balance two objectives)
  3. Include exploration formulations (test excipients you expect to
     be suboptimal, to confirm or refute that expectation)
  4. Span the full concentration range for each excipient, not just
     the "obvious" mid-range
```

**Layer 2 — Domain physics (transferable within domain family):**

Rewritten from independent pharma-formulation literature (Arakawa & Timasheff
preferential-exclusion framework; standard biologics excipient-selection practice) rather
than paraphrasing `excipient_oracle.py`'s code comments — the original version was found to
closely mirror the oracle's own generative logic (near-verbatim restatement of internal
comments), which meant the L1 "correct mechanism" condition was partly testing whether the
LLM could follow the oracle's answer key rather than whether it has genuine domain knowledge.
See `llm_warmstart.py`'s `LAYER2_AGGREGATION` / `LAYER2_OXIDATION` / `LAYER2_WRONG` for the
current text.

**Layer 3 — Specific materials (project-specific, swappable):**
```
Available excipients (choose one from each category):
  Amino acids: arginine(1-25mM), proline(1-25mM), glycine(1-25mM),
    methionine(0.1-3mM), histidine(1-10mM)
  Sugars: sucrose(1-100mM), trehalose(1-100mM), sorbitol(1-50mM), mannitol(1-50mM)
  Surfactants: polysorbate80(0.1-1%), polysorbate20(0.1-1%), poloxamer188(0.1-1%)
  EDTA: yes or no (0.01%)
```

For cross-domain (coatings), only Layers 2 and 3 change. Layer 1 is identical.

**Output schema**: each candidate's `targets` field uses single-letter codes (`T`/`K`/`V` for
Tm/kD/viscosity) instead of full names, and `tradeoff` is capped at 10 words — both decoded/
enforced in `parse_llm_formulations`. This was a deliberate token-budget fix: the original
schema (full objective names, uncapped tradeoff text) caused the model's JSON response to be
truncated mid-object before completing all 25 candidates, which silently failed to parse and
triggered mock fallback on 100% of real-LLM warm-start calls during the first live Phase 2
run. `num_predict` was also raised from 2048 to 4096 as a backstop. `targets`/`tradeoff` are
now actually retained in the parsed formulation dict (previously requested from the LLM then
discarded).

### Metrics

| Metric | What it measures | Source |
|--------|-----------------|--------|
| Hypervolume trajectory | Pareto front quality over time | pymoo HV indicator, fixed reference point |
| Final HV | Overall Pareto front quality | Last batch's HV |
| IGD | Distance to true Pareto front (oracle ground truth) | pymoo IGD against oracle's full pool Pareto front |
| Experiments to 80%/90%/95% of best HV | Sample efficiency | Interpolated from HV trajectory |
| Pareto front size | Diversity of solutions | Non-dominated sort count |
| Stagnant batches | Wasted experiment rounds | Count of batches with zero HV improvement |

**Note (post-Phase-1):** experiments-to-90%-HV showed a ceiling effect at budget=30 in two of
four protein/prior cells (most seeds never reach 90% of true best HV, so both conditions
censor at the budget and the metric loses variance). Experiments-to-70%-HV did not show this
problem and is used as the primary sample-efficiency metric instead (see Statistical testing).

### Statistical testing

**Updated after Phase 1 (mock LLM, rewritten prompt) + power calculation (`power_calc.py`),
run against the actual Phase 1 effect sizes:**

- Effect sizes on **final HV** (`mo_ls_na_egbo` vs `mo_egbo_novelty`) were d=0.01-0.24 across
  the four protein x prior cells — detecting these at 80% power would need 284-2731 seeds
  per group (Bonferroni-corrected over 6 comparisons). Not achievable at any realistic budget.
  **Final HV is demoted to a descriptive/secondary metric — no significance gate.**
- Effect sizes on **experiments-to-90%-HV** were degenerate in two of four cells (d=0.000):
  most seeds never reach 90% of the oracle's true best HV within budget=30, so both conditions
  pile up at the censoring ceiling and the metric loses variance. **Experiments-to-90%-HV is
  not usable as a primary metric at this budget** — either the budget needs to increase until
  90% HV is reachable, or the threshold needs to drop.
- Effect sizes on **experiments-to-70%-HV** were the most tractable: d=-0.84 in the best cell
  (mAb_oxidation/blank, req. n≈24-37/group) down to d=-0.22 in the worst (req. n≈325-501/group).
  This is the only metric where the observed mock effect is large enough to be worth powering
  a real study around.
- **Primary claim (confirmatory, Holm-corrected, p<0.05)**: `mo_ls_na_egbo` reaches 70% of
  best-achievable HV in significantly fewer experiments than `mo_egbo_novelty`, across both
  protein profiles with correct (L1) prior. This replaces the final-HV primary claim.
- **Headline secondary comparison**: `mo_ls_na_egbo` vs `mo_egbo_real` (the strongest existing
  baseline per README) on the same primary metric — reported with effect size + CI, not gated
  on significance, since a reviewer will ask for this comparison regardless of whether the
  novelty-EGBO claim holds.
- **All other reported metrics (final HV, IGD, experiments-to-80%/90%-HV, Pareto front size,
  stagnant batches) and all other conditions (`mo_egbo_real`, `mo_llm_candidate_gen`, `mo_ls_egbo`,
  robustness/blank/wrong-prior comparisons) are secondary/exploratory**: report mean ± std,
  effect size (Cliff's delta), and CI, without a p<0.05 gate. Splitting the confirmatory
  claim across superiority (correct prior), non-inferiority (blank prior), and robustness
  (wrong prior) tests would require separate power budgets for each — the seed budget here
  only supports one confirmatory claim.
- Seed count for Phase 2 should be re-derived from `power_calc.py` run against
  experiments-to-70%-HV once Phase 1 is re-run with `--w_nov` fixed at its final value (the
  0.1-0.3 sweep was itself underpowered at n=15 and did not clearly replicate the "0.1 is the
  sweet spot" finding from the original Aqeeli-based prompt).
- Wilcoxon signed-rank test for pairwise comparisons (paired by seed where initializations are
  shared; treat as approximately paired where `mo_ls_na_egbo`'s LLM-selected init differs from
  the shared LHS init used by other conditions)
- Holm-Bonferroni correction across the primary claim's comparisons only (2 protein profiles);
  secondary metrics reported without correction, consistent with their exploratory status above

## Compute/Resource Estimate

**Updated with measured (not assumed) latency**, per the model-selection benchmark above.
The original ~30s/call estimate was off by more than an order of magnitude — real 72B calls
on this hardware measured 470-555s each, and even qwen3:32b (the selected model) measured
327s/call.

### LLM calls
- **LS-NA-EGBO**: 1 call per seed (warm-start only), at ~327s/call (`qwen3:32b`, measured).
  Phase 2 (5 core conditions, 25 seeds × 2 proteins × 3 priors): 150 warm-start calls ≈ 13.6
  hours of LLM time alone if run fully sequentially. This does not include `mo_llm_candidate_gen`/
  `mo_llm_existing`, which are deferred to a separate follow-up run (see grilling decisions).
- **LLM-as-candidate-generator / existing strategy_mo_llm**: 1 call per batch per seed — at measured per-call
  latency this is proportionally far more expensive than the original ~4hr estimate assumed;
  re-estimate before running either condition.
- **Mock mode**: All conditions runnable without Ollama for debugging.

### Compute
- GP fitting + EGBO acquisition: acquisition scoring was vectorised (single batched
  `acq_fn` call instead of one Python-loop call per candidate) in
  `strategy_mo_egbo_novelty` — this was the dominant non-LLM cost per batch; a full
  budget=40 mock campaign (6 EGBO batches) now completes in ~48s.
- Phase 1 (reduced space, 20 seeds, 3 budgets): ~30 min total (no LLM)
- Phase 2 (full space, 25 seeds, 5 core conditions, with LLM): dominated by the ~13.6hr LLM
  warm-start time above; EGBO-stage compute is comparatively minor post-vectorisation
- Phase 3 (Ada coatings, 20 seeds): not yet re-estimated with qwen3:32b
- **Total estimated runtime**: re-derive from the measured 327s/call warm-start latency and
  the vectorised EGBO stage, not the original ~30s/call assumption

### Execution target
- Default machine (worker-0) is sufficient — no HPC needed
- All computation is CPU-bound (GP fitting) plus local LLM inference (user's Spark GPU)
- Memory: <4 GB (oracle pool is 500 × 16D, GPs are small)

## Success Criteria

**Revised after Phase 1 + power calculation** (see Statistical testing above): final HV
turned out to be statistically unresolvable at any practical seed count (req. n up to
~2700/group), and experiments-to-90%-HV is degenerate at budget=30 (ceiling effect). The
primary claim is retargeted to experiments-to-70%-HV, the only metric where Phase 1's mock
effect sizes were large enough to power a real study. All other criteria below are now
secondary/exploratory (effect size + CI, no significance gate) rather than independently
confirmatory — running superiority, non-inferiority, and robustness tests as separate
confirmatory claims off one seed budget was not statistically supportable.

1. **Primary (confirmatory)**: `mo_ls_na_egbo` reaches 70% of best-achievable HV in
   significantly fewer experiments than `mo_egbo_novelty`, at budget=30+ on the full
   14-excipient space, p<0.05 (Wilcoxon, Holm-corrected across the 2 protein profiles),
   with correct (L1) prior. Seed count to be set from `power_calc.py` run on
   experiments-to-70%-HV, not fixed at 15.
2. **Headline secondary comparison**: `mo_ls_na_egbo` vs `mo_egbo_real` (the strongest
   existing baseline) on the same primary metric — reported as effect size + CI.
3. **Sample efficiency (secondary)**: `mo_ls_na_egbo` reaches 80%/90% of best-achievable HV
   in fewer experiments than `mo_egbo_novelty`, reported descriptively; not gated on
   significance given the ceiling-effect risk observed in Phase 1.
4. **Final HV (secondary)**: reported as mean ± std and effect size, not gated — Phase 1
   showed this metric's effect size is too small to resolve at any realistic n.
5. **Generalisability (secondary)**: same architecture (no code changes, only prompt
   changes) achieves non-inferior performance to EGBO on Ada coatings data.
6. **Blank-prior robustness (secondary)**: `mo_ls_na_egbo` with blank prior does not show a
   large effect-size regression vs `mo_egbo_novelty` — reported descriptively, not as a gated
   non-inferiority test (that requires its own margin and power budget, which this study
   does not allocate).
7. **Wrong-prior robustness (secondary)**: `mo_ls_na_egbo` with wrong mechanism prior
   (aggregation prior on oxidation-prone protein) still outperforms `mo_random` by effect
   size, and does not show a large effect-size regression vs `mo_egbo_novelty` — i.e. the
   novelty-aware EGBO stage should visibly recover from a misleading warm-start, reported
   descriptively rather than as a gated test.
5. **Wrong-prior robustness**: `mo_ls_na_egbo` with wrong mechanism prior (aggregation prior on oxidation-prone protein) still outperforms random, and the novelty-aware EGBO stage recovers from the LLM's misleading warm-start.

## Deliverables

1. **Code**: `llm_warmstart.py`, `novelty_selection.py`, `strategy_ls_na_egbo.py`, `strategy_llm_candidate_gen.py`, `run_benchmark.py` — all saved to `/mnt/results/`
2. **Results CSV**: `benchmark_summary.csv` with all conditions × phases × metrics
3. **Figures** (PNG):
   - HV trajectory by condition (line plot with confidence bands)
   - Experiments-to-90%-HV bar chart by condition
   - Pareto front visualization (3D scatter for Tm/kD/viscosity) for best seed per condition
   - Sample efficiency curve (HV vs experiment count) for the key comparison
4. **Notebook**: Complete reproducible record in `/mnt/results/execution_trace/worker-0.ipynb`
5. **Report** (if results warrant): `/mnt/results/report_llm_seeded_egbo.md` with methods, results, key figures, and references

## Assumptions

1. `qwen3:32b` (switched from qwen2.5:72b-instruct, see Model selection above) is capable of
   generating diverse, mechanistically-reasonable formulations when prompted with the
   excipient catalogue and roles. This is supported by the NeurIPS 2025 finding that
   reasoning LLMs excel on LLM-comprehensible representations, and by the qwen3:32b smoke
   test (25/25 candidates parsed, full excipient/sugar/surfactant diversity) — but formulation
   *quality*, not just successful parsing, has not yet been independently evaluated at scale.
2. The synthetic oracle's dose-response curves are realistic enough that warm-start performance on the oracle predicts warm-start performance on real experiments. This is an inherent limitation of any synthetic benchmark.
3. The novelty-aware selection weight (`w_acq`/`w_nov`) transfers to the formulation domain. Tested in Phase 1 with a sensitivity sweep over `w_nov` in {0.0, 0.1, 0.15, 0.2, 0.3}; the sweep was itself underpowered at n=15 and did not clearly replicate "0.1 is the sweet spot" — see Statistical testing.
4. BoTorch and pymoo are available in the execution environment (they're imported in the existing `strategy_mo_egbo_real` code, suggesting they were installed at some point).
5. The user's local hardware (NVIDIA GB10 / DGX-Spark-class, unified memory) is available for LLM calls during the benchmark runs. Real-LLM latency is meaningfully hardware-bound on this system (measured ~4.6 tok/s for 72B, ~1.7x faster for qwen3:32b) — budget wall-clock accordingly, not against the original ~30s/call estimate. Mock mode can validate the architecture without Ollama.
