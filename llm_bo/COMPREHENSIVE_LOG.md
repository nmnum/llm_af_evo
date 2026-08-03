# Comprehensive Log: llm_bo

**Compiled**: 2026-08-03, from a read-only interpretation pass over the existing codebase and results directories (no new experiments were run to produce this log beyond re-executing the existing `analyse_llm_logs.py` analysis script against already-generated result logs).

---

## 1. Relationship to `sdl_adaptive`

`llm_bo` is a separate, later, architecturally distinct project from `sdl_adaptive` (see `sdl_adaptive/COMPREHENSIVE_LOG.md`). Where `sdl_adaptive`'s controllers (A/B/C/C-evo/D) had the LLM tune hyperparameters, pick among discrete strategies, rewrite optimiser code, or route between strategies — and found EGBO (hand-coded evolutionary + GP Bayesian optimisation) beating all of those LLM-driven conditions — `llm_bo` tests a different mechanism entirely: **using the LLM to rank a pre-generated candidate pool with injected domain knowledge**, rather than having it choose/tune/write the search strategy itself.

## 2. Core mechanism (`posthoc_llm_comparison*.py`)

Architecture (from the module docstring, describing the change from an earlier, much slower v1):

- **GP+evolutionary candidate generation**: a GP+evolutionary-candidates pool of proposals is generated per batch (same underlying machinery as `sdl_adaptive`'s EGBO).
- **LLM ranks, doesn't generate**: the LLM is shown the top-20 pool candidates (by index) plus **domain-specific prior knowledge text** (e.g. for `pareto_20210112`: physical descriptions of what concentration/temperature/flow_rate/pressure do, and known interactions between them) and selects 4 by index. This is a much cheaper LLM call (a handful of integers, not 30 raw coordinate vectors) — the original v1 architecture generated raw coordinates directly and snapped them to the oracle grid, which wasted proposals and took ~18 hours of inference; the pool-ranking redesign cut this to ~2 hours.
- **Caching**: identical (dataset, observation-history-hash) combos reuse the same LLM response — early batches across seeds often share histories since they start from the same init. Estimated cache hit rate 30–50% early in a campaign.
- **Parallelism**: seeds run in parallel (`--n_workers`, default 4).
- **Per-seed JSON logs**: `results_dir/<dataset>/<condition>/seed_NNN.json`, containing auc, final value, and per-decision detail (step, queried_x, queried_y, llm_candidates_shown, llm_selected_indices, gp_scores, cache_hit).

## 3. Conditions

- **`egbo`** — the EGBO baseline (evolutionary candidates + GP-UCB), same reference strategy as in `sdl_adaptive`.
- **`llm_bo`** — pool-ranking LLM: GP+evo pool → LLM ranks (using domain-knowledge prior text) → selects 4. The pool shown to the LLM includes 8 random "padding" candidates mixed in alongside the GP+evo ones.
- **`llm_nopad`** — identical to `llm_bo`, but the 8 random padding candidates are excluded from the pool shown to the LLM. This was designed as an explicit ablation to test whether random padding (extra candidate diversity) was responsible for any advantage `llm_bo` showed over `egbo`.
- **`random`** — pure random search floor.

## 4. Why `llm_bo` (padded) was dropped in favour of `llm_nopad`

Only one run — `results_posthoc_llm_v2` — contains both `llm_bo` and `llm_nopad` side by side (3 datasets: `coatings`, `hartmann6`, `pareto_20210112`, n=20 repeats). Re-running `analyse_llm_logs.py` against it gives:

| Dataset | egbo | llm_bo (padded) | llm_nopad (unpadded) | random |
|---|---|---|---|---|
| coatings | 0.888 | 0.903 | **0.907** | 0.882 |
| hartmann6 | 0.813 | 0.797 | **0.812** | 0.751 |
| pareto_20210112 | 0.703 | 0.718 | **0.733** | 0.671 |

**Paired t-test, `llm_nopad` vs `llm_bo` (the padding-effect test) — none significant:**

| Dataset | Δ (llm_nopad − llm_bo) | p |
|---|---|---|
| coatings | +0.004 | 0.60 |
| hartmann6 | +0.015 | 0.37 |
| pareto_20210112 | +0.015 | 0.47 |

`llm_nopad` was consistently ≥ `llm_bo` across all three datasets tested — removing the random padding candidates never hurt performance, and numerically helped slightly, though the differences don't reach significance. **Conclusion: the random padding was not responsible for `llm_bo`'s apparent edge over EGBO — if anything, the pool-ranking LLM performs marginally better without it.** This closed the padding question, and `llm_bo` (padded) was left out of all subsequent, larger runs (`results_posthoc_llm_v3`, `results_posthoc_llm_v4` — both confirmed to contain only `egbo`, `llm_nopad`, `random`, no `llm_bo` directory). `llm_nopad` is the canonical/primary LLM condition referenced in later results and going forward.

## 5. `llm_nopad` vs EGBO — cross-dataset results

### `results_posthoc_llm_v2` (3 datasets, n=20)

| Dataset | egbo | llm_nopad | random | Δ vs egbo | p |
|---|---|---|---|---|---|
| coatings | 0.888 | 0.907 | 0.882 | +0.019 | 0.078 |
| hartmann6 | 0.813 | 0.812 | 0.751 | −0.001 | 0.97 |
| pareto_20210112 | 0.703 | 0.733 | 0.671 | +0.030 | 0.27 |

No dataset reaches p<0.05 at this scale/dataset selection.

### `results_posthoc_llm_v3` (5 datasets — adds `hartmann3`, `pareto_20201218` — n=20; no `llm_bo` condition present)

| Dataset | egbo | llm_nopad | random | Δ vs egbo | p |
|---|---|---|---|---|---|
| coatings | 0.890 | 0.899 | 0.882 | +0.009 | 0.46 |
| hartmann3 | 0.917 | 0.942 | 0.870 | +0.025 | **0.0014** |
| hartmann6 | 0.815 | 0.809 | 0.751 | −0.006 | 0.80 |
| pareto_20201218 | 0.579 | 0.667 | 0.658 | +0.088 | 0.056 (borderline) |
| pareto_20210112 | 0.709 | 0.732 | 0.671 | +0.023 | 0.41 |

`results_posthoc_llm_v4` has the same condition set (`egbo`, `llm_nopad`, `random`; no `llm_bo`) as v3 — treated as a repeat/extension of the v3 design rather than a new architecture.

**Only `hartmann3` reaches conventional significance (p=0.0014) for `llm_nopad` beating `egbo`.** `pareto_20201218` has the largest raw effect size (+0.088) but is borderline (p=0.056). The other three datasets (coatings, hartmann6, pareto_20210112) show small, non-significant deltas — sometimes even slightly negative (hartmann6).

## 6. Timing pattern: when the LLM's advantage shows up (v2 seed-level analysis)

`analyse_llm_logs.py`'s seed-level win-count breakdown (`llm_nopad` vs `egbo`, out of 20 seeds each) shows the LLM's advantage is not a uniform "converges faster" story — it is dataset-dependent in *when* it appears:

| Dataset | llm_nopad wins early | llm_nopad wins late | egbo wins | Tied | Pattern |
|---|---|---|---|---|---|
| coatings | 10/20 | 0/20 | 5/20 | 5/20 | Early-campaign advantage |
| hartmann6 | 2/20 | 7/20 | 8/20 | 3/20 | Late-campaign advantage |
| pareto_20210112 | 8/20 | 1/20 | 6/20 | 5/20 | Early-campaign advantage |

Consistent with this, `results_posthoc_llm_v3`'s `hartmann3` breakdown also shows a **late-campaign** advantage pattern (llm_nopad wins early 2/20, late 10/20, egbo wins 1/20, tied 7/20) despite hartmann3 being the dataset with the strongest overall significant win — the win accrues late in the campaign there, not early.

**Interpretation**: on the smoother/simpler landscapes here (coatings, pareto_20210112), the LLM's domain-knowledge-informed ranking helps it exploit good regions earlier than EGBO's acquisition-driven search; on harder landscapes (hartmann6, hartmann3) the benefit shows up later, once enough of the space has been explored for the domain-knowledge prior to differentiate remaining candidates usefully.

## 7. Overall takeaway

Unlike `sdl_adaptive`, where every LLM-driven strategy-selection/tuning/code-rewriting approach underperformed hand-coded EGBO, `llm_bo`'s candidate-pool-ranking mechanism (with real domain-knowledge priors injected into the prompt) shows a **directionally favourable, and in one case (hartmann3) clearly significant, result for the LLM condition over EGBO** — though the effect is landscape-dependent and only reaches significance in 1 of 5 datasets tested at n=20 (with `pareto_20201218` borderline). The random-padding ablation (`llm_bo` vs `llm_nopad`) confirmed the advantage, where present, isn't an artefact of extra random candidate diversity — `llm_nopad` performs at least as well as `llm_bo` throughout, which is why later runs (v3, v4) dropped the padded condition entirely and standardised on `llm_nopad`.

The key methodological difference between the two projects, worth carrying forward: giving an LLM **real, stated domain knowledge about the physical system** and asking it to rank a small set of already-good candidates appears to be a more promising use of LLM judgment in this benchmark family than asking it to tune hyperparameters, pick among abstract strategy names, or write optimiser code from scratch.
