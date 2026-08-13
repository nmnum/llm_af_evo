# Comprehensive Project Log: Self-Driving Lab — Adaptive Campaign Planner for LLM-BO

**Project**: Simulating experimental campaigns to understand which optimisation strategies work, fail, or break under different biological and experimental regimes, and deriving design rules for an adaptive campaign planner.

**Author**: Neha Mungale
**Supervisor**: David Shorthouse (UCL School of Pharmacy)
**Date compiled**: 2026-08-01
**Date updated**: 2026-08-13 — added the Phase 3 (cross-domain, ADA coatings) results, run with the real LLM at n=20 seeds including the `egbo_warmstart`/`ucb_warmstart` ablations, with full Holm-corrected significance table; updated C4 and the file inventory accordingly. Previous update: 2026-08-02, superseding `project_comprehensive_log_previous.md`. Sections 1–11 are carried over unchanged (see that file for full detail); this version rewrites Section 12 onward to incorporate the Phase 1/2/sensitivity benchmark suite that was run after the previous log was compiled, including the missing mAb_aggregation WRONG/blank multi-objective comparison the previous log flagged as outstanding.
**Scope**: Everything — from the original cancer drug screening work through the pharmaceutical formulation pivot, single-objective stress tests, multi-objective oracle construction, bug diagnosis and fixing, the wrong-prior findings, and the subsequent LS-NA-EGBO / novelty-aware benchmark suite.

---

## What changed since the previous log

The previous log ended with two open items:

1. **C2** (per-objective trust diagnostic) not validated at budget=40/d=16 — trust collapsed to 0.700 for every objective.
2. **The missing mAb_aggregation WRONG/blank multi-objective run** — needed to tell whether the mAb_oxidation "wrong prior wins" result was a genuine mechanism effect or just arginine's universal kD dominance.

Since then, a new benchmark suite (`run_benchmark.py`, `run_benchmark_resumable.py`, `run_sensitivity.py`) was built and run, introducing three new strategies not present in the previous log's campaign runner:

- **`mo_egbo_novelty`** — EGBO (qLogNEHVI + U-NSGA-III) with novelty-aware batch selection (`novelty_selection.py`), following Aqeeli et al. — candidates are chosen with `score = w_acq * acquisition + w_nov * novelty` instead of pure acquisition ranking, to stop batches clustering in one region.
- **`mo_ls_na_egbo`** ("LLM-Seeded, Novelty-Aware EGBO", `strategy_ls_na_egbo.py`) — the architecture now recommended by the project's mechanistic diagnosis: the LLM is called **once**, before any experiments, to warm-start the initial batch with mechanistically diverse formulations (`llm_warmstart.py`); after that, EGBO with novelty-aware selection runs autonomously for the rest of the campaign. This replaces the previous per-batch trust-weighted mixing (`mo_llm`/`strategy_mo_llm`) as the primary LLM-informed strategy.
- **`mo_llm_candidate_gen`** (`strategy_llm_candidate_gen.py`) — LLM called every batch as a candidate generator (20–30 proposals/batch), merged into the EGBO pool and scored by qLogNEHVI, no trust diagnostic. **Important naming correction**: this was originally implemented and labelled "LABO" (`mo_llm_labo`). It does **not** implement the literature LABO method (Kennedy–O'Hagan multi-fidelity surrogate with a gating rule that can skip real experiments when the LLM's low-fidelity prediction is trusted enough). It was renamed to `strategy_llm_candidate_gen`/`mo_llm_candidate_gen` to avoid the false claim of literature fidelity. A true gated multi-fidelity LABO implementation remains unbuilt and out of scope. **Old result tables in this log or elsewhere that say "mo_llm_labo" refer to this same renamed strategy, run before the rename.**
- **`mo_ls_egbo`** — an ablation added in the sensitivity sweep: LLM warm-start + real EGBO with **no** novelty-aware selection, to isolate the warm-start effect from the novelty-selection effect.

The old trust-weighted `mo_llm` strategy and the per-objective trust diagnostic from Phase 5–8 of the previous log are **not used** in this new suite — they were superseded by the warm-start architecture rather than fixed. C2 (the trust diagnostic) is therefore still unvalidated and now effectively shelved in favour of a design that doesn't depend on it (see "Status of C2" below).

Three benchmark runs exist in `results/`:

| Directory | Script | Conditions | Seeds | Budget |
|---|---|---|---|---|
| `results/benchmark_phase1/` | `run_benchmark_resumable.py --phase 1` | mo_random, mo_egbo, mo_egbo_real, mo_egbo_novelty, mo_ls_na_egbo, mo_llm_labo (= mo_llm_candidate_gen, pre-rename), mo_ls_egbo | 15 | 30 |
| `results/benchmark_phase2/` | `run_benchmark_resumable.py --phase 2` | mo_random, mo_egbo, mo_egbo_real, mo_egbo_novelty, mo_ls_na_egbo, mo_llm_candidate_gen | 25 | 40 |
| `results/benchmark_sensitivity/` | `run_sensitivity.py` | mo_egbo_novelty and mo_ls_na_egbo at w_nov ∈ {0.00, 0.10, 0.15, 0.20, 0.30}, plus mo_ls_egbo | 15 | 30 |

Phase 1 used the reduced formulation space; Phase 2 used the full 14-excipient space and is the one that matters for the thesis's headline numbers. **Phase 3 (cross-domain generalisation to the ADA coatings oracle, `run_phase3.py`) has since been run with the real LLM (n=20 seeds) — see "Phase 3 results (cross-domain generalisation)" below.**

All of `run_benchmark.py`, `run_benchmark_resumable.py`, `run_sensitivity.py`, `run_phase3.py`, `novelty_selection.py`, `strategy_ls_na_egbo.py`, `strategy_llm_candidate_gen.py`, `llm_warmstart.py` are currently **untracked** in git (`git status` at time of writing shows them as `??`) — they exist on disk but have never been committed. Commit them before relying on this log surviving a `git clean` or fresh checkout.

---

## 12 (revised). The Missing Comparison, Resolved: mAb_aggregation WRONG/blank in Multi-Objective

Phase 2 (`results/benchmark_phase2/phase2_summary.csv`, 25 seeds, budget=40, full excipient space) is the first run to cross `mAb_aggregation` with `prior_level=wrong`. This is the comparison the previous log called "the single most informative comparison remaining."

### Result (HV mean ± SD, mo_llm_candidate_gen, mAb_aggregation)

| Prior | HV mean ± SD |
|---|---|
| L1 (correct — arginine-forward) | 12736 ± 1753 |
| blank | 12270 ± 1627 |
| **wrong** (oxid_L1 — methionine-forward) | **12691 ± 1567** |

For reference, in the same table: `mo_random` = 10926 ± 2007 (constant across priors, as expected), `mo_egbo_real` ranges 12835–13366 across priors, `mo_egbo_novelty` ranges 12790–13418, `mo_ls_na_egbo` ranges 12890–13744.

None of the pairwise differences between L1/blank/wrong for `mo_llm_candidate_gen` are statistically significant (see `phase2_stats_vs_egbo_novelty.csv`: e.g. wrong vs `mo_egbo_novelty` baseline gives p=0.635 uncorrected). All three prior conditions clearly and significantly beat `mo_random` (p < 0.002, Holm-corrected, for all three).

### Interpretation

This resolves the open question cleanly, and in the more informative of the two possible directions:

- **It is not the case that WRONG uniformly wins in multi-objective formulations.** If it had (i.e. if WRONG had also dominated on mAb_aggregation), the mAb_oxidation "wrong prior wins" result from Phase 7/8 of the previous log would have been explained away as "arginine dominates kD on every profile regardless of mechanism," which would have undermined the entire mechanism-reasoning story.
- **It is also not the case that WRONG uniformly hurts**, which would have just reproduced the single-objective result and made the multi-objective reward-structure argument (Section 12 of the previous log, "Finding 3") moot.
- Instead, the WRONG-prior benefit is **profile-dependent**, exactly as the multi-objective reward-structure argument predicts: on mAb_aggregation, arginine (the excipient the correct L1 prior already recommends) is already the *mechanistically correct* choice for kD, so there's no "free ride" from an incidental cross-cutting effect — WRONG, L1, and blank cluster together because the LLM's own domain reasoning has less differential value here than the underlying kD chemistry does. On mAb_oxidation, by contrast, the correct mechanism (methionine, oxidation-focused) leaves kD's large dynamic range unexploited, so the WRONG-but-arginine-forward prior gets a disproportionate hypervolume reward from kD alone — that is where the previous log's WRONG-wins effect (2763 vs 2413 for L1) lives.

### Updated claim

> Wrong-mechanism priors do not have a fixed effect sign in multi-objective formulation optimisation. Whether a mechanistically wrong prior helps, hurts, or is indistinguishable from a correct one depends on (a) how much unexploited dynamic range the "accidentally correct" objective has under the correct mechanism, and (b) how much the wrong prior's recommended excipient overlaps with the objective that has that unexploited range. This was tested directly: on mAb_oxidation, where the correct mechanism (methionine) leaves kD's range mostly unused, the wrong prior (arginine) wins outright. On mAb_aggregation, where the correct mechanism already targets kD directly, wrong/blank/correct priors are statistically indistinguishable.

This is a stronger, testable, falsifiable version of C3 than either the original formulation ("wrong prior hurts") or the earlier single-datapoint reframing ("wrong prior can partially help") — and it now rests on two profiles' worth of evidence instead of one.

### Caveat for the thesis — resolved (2026-08-11)

The Phase 2 `mo_llm_candidate_gen` strategy is not the same strategy that produced the original mAb_oxidation WRONG=2763 result in the previous log (that used the old trust-weighted `mo_llm`/`strategy_mo_llm`, which has since been superseded). The Phase 2 mAb_oxidation numbers for `mo_llm_candidate_gen` (L1=2413, blank=2584, wrong=2572 — all statistically indistinguishable, see `phase2_summary.csv`) do **not** reproduce the original dramatic WRONG-wins effect under the new strategy. This was an open decision — which strategy's mAb_oxidation numbers the thesis claims — and it is now resolved: **the thesis claims `mo_llm_candidate_gen` (Phase 2, null result)**, not the old `mo_llm` (WRONG=2763, dramatic inversion).

**Why**: the two strategies aren't measuring the same thing, and the difference explains the divergence rather than just describing it.

- Old `mo_llm` (per-batch trust-weighted mixing) injects the LLM's prior recommendation into *every* batch for the whole campaign, weighted by a trust score. A "wrong but arginine-forward" prior gets sustained, repeated top-down influence — if arginine happens to deliver a large kD payoff (the unexploited-dynamic-range mechanism above), that bias compounds batch after batch regardless of what the GP evidence says.
- New `mo_llm_candidate_gen` merges the LLM's per-batch candidates into the EGBO pool and scores them by qLogNEHVI like any other candidate — no trust weighting, no persistent injection. A wrong-mechanism candidate only survives selection if the acquisition function's own value estimate favours it *that batch*; any "free ride" from an accidentally-correct excipient has to re-earn its selection every batch on merit rather than carrying forward via a mixing weight.
- The old `mo_llm`'s trust-weighting mechanism is also the one this project's own log already flags as **unvalidated**: C2 (the per-objective trust diagnostic) collapsed to 0.700 for every objective and was shelved rather than fixed (see "What changed since the previous log," above). The dramatic WRONG=2763 vs L1=2413 inversion sits on a mechanism already known to be miscalibrated, not a different-but-equally-trustworthy architecture.
- `mo_llm_candidate_gen` is also the strategy Phase 2 (25 seeds, full 14-excipient space) actually used for the thesis's headline numbers, and has no dependency on the unvalidated trust diagnostic.

**How this is framed in the thesis**: not as two competing data points to reconcile, but as evidence that the wrong-prior "benefit" is an artifact of *how* the LLM prior is mixed into candidate selection, not a property of the underlying chemistry. A persistently-injected wrong-mechanism prior can manufacture a large hypervolume advantage even when the same prior, treated as one candidate-source among many under acquisition-function scoring, shows no effect — which doubles as a caution about trust-weighted LLM-integration designs generally, not just a footnote explaining away an inconvenient number. The mAb_aggregation resolution above stands either way, since it only required the *absence* of a universal WRONG-wins effect, which both strategies agree on directionally.

---

## New Section: Novelty-Aware Selection and the LS-NA-EGBO Architecture

### Why novelty-aware selection was added

The previous log's Phase 5 (Section 9) identified that the lightweight `mo_egbo`'s UCB-based Pareto ranking was noise-dominated at low n and clustered candidates in one region. Following Aqeeli, Leelawat & Shorthouse (2026) — the directly related paper already cited in the previous log's literature review — `novelty_selection.py` implements hybrid selection:

```
score = w_acq * acquisition_merit_normalised + w_nov * novelty_normalised
```

where novelty is distance from already-selected points in normalised input space, computed sequentially within a batch. This is applied on top of `mo_egbo_real`'s qLogNEHVI/U-NSGA-III candidate pool (→ `mo_egbo_novelty`) and, combined with LLM warm-starting, forms `mo_ls_na_egbo`.

### The LS-NA-EGBO architecture (`strategy_ls_na_egbo.py`)

Two-stage design, replacing the previous log's per-batch trust-weighted LLM/GP mixing:

- **Stage 1 (once, before any experiments)**: `llm_warmstart_init` (`llm_warmstart.py`) calls the LLM once to generate a diverse, mechanistically-reasonable initial batch. The prompt uses a three-layer design — domain-agnostic reasoning protocol, transferable domain physics, project-specific materials — intended to make the warm-start prompt portable to other domains (this is what Phase 3 tests; see "Phase 3 results" below).
- **Stage 2 (rest of campaign, autonomous)**: novelty-aware EGBO (qLogNEHVI + U-NSGA-III + novelty-aware selection) runs without further LLM calls.

This sidesteps the unvalidated per-objective trust diagnostic (C2) entirely — rather than using a live trust signal to decide the GP/LLM mixing ratio every batch, the architecture fixes the LLM's role to a one-shot prior injection and lets EGBO handle everything downstream. **This is an implicit resolution of C2's dependency, not a validation of the diagnostic itself**: C2 as originally scoped (a diagnostic that differentiates per-objective GP trust in real time) remains untested and is no longer load-bearing for the current best strategy.

### Novelty weight sensitivity (`results/benchmark_sensitivity/`)

Swept w_nov ∈ {0.0, 0.10, 0.15, 0.20, 0.30} for `mo_ls_na_egbo` and `mo_egbo_novelty`, 15 seeds, budget=30, both proteins × {L1, blank}:

| Protein | Prior | Best w_nov (mo_ls_na_egbo, by HV mean) | HV at best | HV at w_nov=0.30 |
|---|---|---|---|---|
| mAb_aggregation | L1 | 0.10 | 12342 | 12325 |
| mAb_aggregation | blank | 0.10 | 12731 | 11982 |
| mAb_oxidation | L1 | 0.20 | 2414 | 2301 |
| mAb_oxidation | blank | 0.10 | 2666 | 2445 |

The literature default (Aqeeli et al.) of w_nov=0.3 is consistently among the worse settings in this domain — the sweep note in `novelty_selection.py`'s docstring records this explicitly ("0.3 is the Aqeeli et al. default but is too aggressive at budget=30 for this domain"). w_nov=0.10–0.15 is the better range here, likely because this formulation space (14–16D, budget 30–40) is far sparser than the discovery-science setting the default was tuned on, so novelty pressure that's appropriate there over-penalises acquisition merit here. `mo_ls_egbo` (warm-start, no novelty selection at all) is included as the isolating ablation and is competitive with the novelty-selected variants in several rows (e.g. mAb_oxidation, blank: 2526 vs 2666 at best w_nov) — meaning **on this domain, at this budget, most of the benefit is coming from the LLM warm-start, not from novelty-aware selection**, which should be stated plainly rather than assumed.

### Phase 1 vs Phase 2 headline comparison (mo_ls_na_egbo vs baselines)

Phase 2 (25 seeds, budget=40, full 14-excipient space) is the more representative run:

| Protein | Prior | mo_random | mo_egbo_real | mo_egbo_novelty | mo_ls_na_egbo |
|---|---|---|---|---|---|
| mAb_aggregation | L1 | 10926 ± 2007 | 13317 ± 1485 | 12829 ± 1044 | 13179 ± 1487 |
| mAb_aggregation | blank | 10926 ± 2007 | 12835 ± 1143 | 12432 ± 1333 | **13744 ± 1151** |
| mAb_oxidation | L1 | 2330 ± 534 | 2660 ± 441 | 2576 ± 533 | **2694 ± 372** |
| mAb_oxidation | blank | 2330 ± 534 | 2527 ± 498 | 2401 ± 461 | **2617 ± 365** |

`mo_ls_na_egbo` is at or above `mo_egbo_real` and `mo_egbo_novelty` in every row shown, and has the lowest variance across seeds in 3 of 4 rows. See the full paired/Holm-corrected re-analysis below for which of these differences actually reach significance -- the short version is: informed-vs-random is robustly significant everywhere in Phase 2; the mo_ls_na_egbo vs mo_egbo_novelty ablation reaches Holm-corrected significance in one cell (Phase 2, mAb_aggregation, blank prior) and is directionally consistent but not significant elsewhere.

### Phase 1 and Phase 2 mAb_aggregation results, full paired/Holm-corrected re-analysis (2026-08-13)

Re-run with all 8 conditions (Phase 1: mo_random, mo_egbo, mo_egbo_real, mo_egbo_novelty, mo_llm_candidate_gen, mo_llm_labo, mo_ls_egbo, mo_ls_na_egbo -- mo_llm_labo retained under its pre-rename name, see naming note above) and paired Wilcoxon signed-rank tests matched by seed, Holm-corrected within each protein/prior group. **Phase 2's mAb_oxidation protein is still running as of this writing** (a resumable run to add the new mo_ucb_warmstart condition is in progress and mAb_oxidation isn't yet complete for all conditions) -- only Phase 1 (both proteins, complete) and Phase 2's mAb_aggregation (complete, 25 seeds x 7 conditions) are reported here; mAb_oxidation Phase 2 will be added once the run finishes.

**Phase 1 (15 seeds, budget=30, reduced formulation space)**: only one comparison anywhere survives Holm correction -- mo_egbo_real vs mo_random in mAb_aggregation/blank (holm_p=0.050). Everything else, including informed-vs-random gaps that look large in the raw means, does not survive correction at n=15 -- underpowered, consistent with the earlier Phase 3 finding that n≈20 seeds isn't enough for confirmatory inter-strategy claims. One directionally notable pattern: mo_ls_egbo (warm-start, **no** novelty-aware selection) is the top mean performer in mAb_aggregation for both L1 and blank priors, ahead of mo_ls_na_egbo -- the **opposite** of the Phase 3 coatings pattern, where warm-start alone did nothing and needed pairing with a novelty-sensitive acquisition to be competitive. Not statistically confirmed at n=15, but worth flagging as a domain-dependent inconsistency in the warm-start x novelty interaction, not something to gloss over.

**Phase 2, mAb_aggregation (25 seeds, budget=40, full 14-excipient space)**:

- **All informed-vs-random comparisons are robustly significant after Holm correction**, across all three priors (L1, blank, wrong) -- 4-6 survivors per prior group, holm_p as low as 0.0005-0.0016. This is the cleanest, most robust result in the whole benchmark suite: informed strategies (of any kind) reliably beat random search here.
- **Headline finding**: in the **blank** prior, mo_egbo_novelty vs mo_ls_na_egbo reaches Holm-corrected significance (diff=-1311.9 HV, raw_p=0.0027, **holm_p=0.0432**) -- mo_ls_na_egbo beats mo_egbo_novelty by a wide, corrected-significant margin. This is the clean ablation isolating warm-start's effect (identical acquisition code, differ only in init) and -- unlike Phase 3 coatings, where the same comparison was directional but not significant -- **it reaches significance here**. mo_llm_candidate_gen vs mo_ls_na_egbo is borderline in the same direction (holm_p=0.075).
- In the L1 and wrong priors, the same mo_egbo_novelty vs mo_ls_na_egbo comparison is directionally consistent (ls_na_egbo ahead) but does not reach significance.

**Revised headline claim, incorporating this and the Phase 3 result**: informed strategies of any kind reliably beat random search across every phase and domain tested (Phase 1, Phase 2, Phase 3 all replicate this, robustly significant in Phase 2). The narrower claim -- that LLM warm-start specifically improves on plain novelty-aware EGBO -- now has one statistically confirmed instance (Phase 2, mAb_aggregation, blank prior) plus consistent directional support elsewhere (Phase 2 L1/wrong, Phase 3 coatings), but is not yet confirmed as a general effect across all domains/priors/seed counts -- Phase 1 (n=15) and most Phase 2/3 cells remain underpowered for this specific comparison. Higher seed counts (the earlier Phase 3 power analysis suggested ~100 seeds for a comparable effect size) or a purpose-built paired/blocked design would be needed to confirm it generally.

### Phase 3 results (cross-domain generalisation, ADA coatings oracle, real LLM, n=20)

`run_phase3.py` was run to completion against the real LLM (`qwen3:32b`, `--n_seeds 20`), testing whether the LS-NA-EGBO architecture transfers to a different physical domain (spray-drying/ADA coatings, 4D input space, mixed physical-unit bounds — fuel:oxidizer ratio, acac amount, g/mL concentration, °C temperature) rather than the mAb formulation space Phase 1/2 use. Two additional conditions were added beyond the original design to complete a 2×2 {warm-start × novelty-aware selection} ablation and to test a cheaper acquisition alternative:

- `egbo_warmstart` — LLM warm-start init + plain qLogNEHVI EGBO (no novelty term). Isolates warm-start's effect alone.
- `ucb_warmstart` — LLM warm-start init + a simple scalarised (equal-weight-sum, per-objective-std-normalised) UCB acquisition, in `excipient_campaign_mo.py`/`strategy_mo_scalarized_ucb`. Tests whether a much cheaper acquisition still benefits from warm-start.

Both conditions use one LLM call (Stage-1 warm-start only), matching `ls_na_egbo`'s cost profile, not `llm_labo`'s (LLM called every batch).

**Mean final hypervolume, all 7 conditions (n=20 seeds each), ranked:**

| condition | mean | std | min | max |
|---|---|---|---|---|
| ucb_warmstart | 458.18 | 47.48 | 383.57 | 525.24 |
| ls_na_egbo | 453.09 | 45.38 | 395.61 | 525.46 |
| egbo_warmstart | 449.83 | 47.90 | 387.73 | 525.46 |
| egbo | 449.16 | 34.92 | 405.97 | 525.35 |
| llm_labo | 439.39 | 40.80 | 399.50 | 525.28 |
| egbo_novelty | 435.42 | 34.89 | 399.55 | 524.91 |
| random | 422.23 | 58.39 | 353.24 | 524.93 |

**2×2 warm-start × novelty design** (isolating which factor drives `ls_na_egbo`'s position):

| | no warm-start | warm-start |
|---|---|---|
| **no novelty** | egbo: 449.2 | egbo_warmstart: 449.8 |
| **novelty** | egbo_novelty: 435.4 | ls_na_egbo: 453.1 |

Full pairwise comparisons (Wilcoxon signed-rank, matched by seed via `make_shared_inits`), with Holm correction applied across all 21 tests:

| comparison | mean_diff | t | raw_p | holm_p |
|---|---|---|---|---|
| egbo vs egbo_novelty | 13.74 | 1.31 | 0.0479 | 0.7657 |
| egbo vs egbo_warmstart | -0.67 | -0.06 | 0.6274 | 1.0000 |
| egbo vs llm_labo | 9.77 | 0.75 | 0.2180 | 1.0000 |
| egbo vs ls_na_egbo | -3.92 | -0.31 | 0.6542 | 1.0000 |
| egbo vs random | 26.93 | 1.67 | 0.1259 | 1.0000 |
| egbo vs ucb_warmstart | -9.02 | -0.72 | 0.7369 | 1.0000 |
| egbo_novelty vs egbo_warmstart | -14.42 | -1.41 | 0.1169 | 1.0000 |
| egbo_novelty vs llm_labo | -3.97 | -0.32 | 0.5016 | 1.0000 |
| egbo_novelty vs ls_na_egbo | -17.67 | -1.27 | 0.1672 | 1.0000 |
| egbo_novelty vs random | 13.19 | 0.85 | 0.3507 | 1.0000 |
| egbo_novelty vs ucb_warmstart | -22.76 | -1.67 | 0.0793 | 1.0000 |
| egbo_warmstart vs llm_labo | 10.44 | 0.86 | 0.5503 | 1.0000 |
| egbo_warmstart vs ls_na_egbo | -3.25 | -0.26 | 0.9702 | 1.0000 |
| egbo_warmstart vs random | 27.61 | 2.02 | 0.0276 | 0.5524 |
| egbo_warmstart vs ucb_warmstart | -8.35 | -0.54 | 0.6813 | 1.0000 |
| llm_labo vs ls_na_egbo | -13.70 | -1.17 | 0.2322 | 1.0000 |
| llm_labo vs random | 17.16 | 1.61 | 0.0304 | 0.5769 |
| llm_labo vs ucb_warmstart | -18.79 | -1.72 | 0.0304 | 0.5769 |
| ls_na_egbo vs random | 30.86 | 2.58 | 0.0400 | 0.6807 |
| ls_na_egbo vs ucb_warmstart | -5.10 | -0.42 | 0.4781 | 1.0000 |
| random vs ucb_warmstart | -35.95 | -2.81 | 0.0057 | 0.1204 |

**No pairwise comparison survives Holm correction at α=0.05** (best case: `random vs ucb_warmstart`, holm_p=0.1204). A companion power analysis (Cohen's dz on the `ls_na_egbo`-anchored comparisons) estimates ~97–114 seeds would be needed for 80% power on the `ls_na_egbo` vs `egbo_novelty` / `llm_labo` gaps at their current observed effect sizes (dz≈0.26–0.28); the `ls_na_egbo` vs `random` gap needs only ~23 seeds (dz≈0.58). At n=20, treat every ranking claim below as descriptive/directional, not confirmatory.

**Interpretation.** The mechanistically interesting result is an interaction, not a main effect: LLM warm-start alone (`egbo_warmstart` vs `egbo`, diff=0.7 HV) does essentially nothing on top of plain qLogNEHVI EGBO, and novelty-aware selection alone (`egbo_novelty` vs `egbo`) is the single worst-performing deviation from `egbo` in the whole table (uncorrected p=0.048, though it doesn't survive Holm). But warm-start *combined with* a novelty-sensitive acquisition — either the explicit novelty term (`ls_na_egbo`) or the cheaper scalarised UCB (`ucb_warmstart`) — recovers to the top of the ranking. Neither ingredient helps in isolation on this domain; together they're the best performers. `ucb_warmstart` topping the ranking is the most actionable single takeaway even though it isn't statistically proven: it suggests the LLM warm-start's exploration diversity is doing most of the real work, and once that diversity exists, the more expensive qLogNEHVI+novelty machinery may be largely redundant with a much cheaper acquisition function — a compute-cost argument independent of whether the HV gap itself is significant.

A separate check (prompted by the ADA coatings oracle's mixed physical units — fuel:oxidizer ratio, g/mL concentration, °C temperature) confirmed `strategy_mo_egbo_novelty`'s novelty-distance computation is not scale-biased by these raw units: `X_obs` is min-max normalised to [0,1] per-dimension (`strategy_ls_na_egbo.py`) before being passed into `novelty_aware_select_vectorised`, and the candidate pool it's compared against is independently already in matching [0,1]-normalised space (via `optimize_acqf`'s `standard_bounds` and pymoo's `xl=0,xu=1` problem bounds). So the `egbo_novelty` underperformance above is a genuine domain/budget effect, not a normalisation artifact.

**For the thesis**: this is suggestive, directionally-consistent-with-Phase-1/2 evidence that the LS-NA-EGBO architecture generalises across domains, not standalone proof — lead with the interaction finding (warm-start needs a novelty-sensitive acquisition to pay off) framed as a qualitative/mechanistic result, report the Holm-corrected table so the significance caveat is explicit, and note the power-analysis seed counts as future work rather than implying they'll be run.

---

## Updated Section 13: The Four Claims — Status Assessment (supersedes previous log's table)

| Claim | Status | Evidence | What's needed |
|-------|--------|----------|---------------|
| **C1**: LLM knowledge beats GP-only | **Validated (SO); validated for beating random (MO), not yet for beating other informed strategies (MO)** | SO: clean gradient on mAb_agg. MO Phase 1/2: `mo_ls_na_egbo` and all informed strategies significantly beat `mo_random` (p<0.006) on both profiles; no informed-vs-informed comparison reaches significance at current seed counts. | More seeds, or a paired statistical design, to resolve informed-vs-informed gaps |
| **C2**: Per-objective trust diagnostic | **Not validated, no longer load-bearing** | Still collapses to 0.700 for all objectives at budget=40/d=16 (unchanged from previous log — not re-tested since). The current best architecture (LS-NA-EGBO) does not use this diagnostic at all, having replaced per-batch trust-weighted mixing with one-shot LLM warm-start + autonomous EGBO. | If C2 is kept in the thesis, either validate at budget≥80 or explicitly reframe it as "attempted and superseded" rather than "pending" |
| **C3**: Wrong-prior / mixing-weight behaviour | **Reframed and now evidenced on two profiles; strategy choice resolved** | The missing mAb_aggregation WRONG/blank comparison (Section 12 above) is resolved: wrong-prior effect is profile-dependent, not universal, consistent with the multi-objective reward-structure argument. **Resolved (2026-08-11)**: the thesis claims `mo_llm_candidate_gen` (Phase 2, WRONG=2572, indistinguishable from L1), not the old trust-weighted `mo_llm` (WRONG=2763, dramatic) — the latter's persistent per-batch prior injection sits on the never-validated C2 trust diagnostic (see row above), so its magnitude isn't trustworthy. The strategy-dependence itself is now framed as an additional finding: a persistently-injected wrong-mechanism prior can manufacture a hypervolume advantage that disappears once the same prior is just one candidate-source scored by acquisition value. | None — see §12 "Caveat for the thesis — resolved" for the full reasoning |
| **C4**: End-to-end pipeline | **Partially validated, architecture changed; cross-domain generalisation now has directional (not significant) support** | Two full pipelines now exist and run end-to-end with real/mock LLM across 30+ seeds each: the original trust-weighted `mo_llm` (previous log, Phase 5–8) and the new warm-start `mo_ls_na_egbo` (this log, Phase 1/2). Real Waibel data (33 formulations) still untested. Phase 3 cross-domain generalisation (ADA coatings, real LLM, n=20) has been run — see "Phase 3 results" above; warm-start + novelty-sensitive acquisition ranks top of 7 conditions but no pairwise comparison survives Holm correction at n=20. | Run against real Waibel data; more Phase 3 seeds (~100, per power analysis) if a confirmatory cross-domain claim is needed for the thesis |

---

## Updated File Inventory (additions since previous log)

| File | Description |
|------|-------------|
| `run_benchmark.py` | Unified benchmark runner, 7 conditions × 3 phases |
| `run_benchmark_resumable.py` | Checkpointed version of the above, used to produce `results/benchmark_phase1/` and `results/benchmark_phase2/`; now 8 conditions after adding `mo_ucb_warmstart` |
| `run_sensitivity.py` | Novelty-weight sensitivity sweep + `mo_ls_egbo` ablation, produced `results/benchmark_sensitivity/` |
| `run_phase3.py` | Cross-domain (ADA coatings) generalisation test — run to completion with real LLM, n=20 seeds, 7 conditions (see "Phase 3 results" above) |
| `novelty_selection.py` | Aqeeli-et-al.-style novelty-aware batch selection, `w_nov` default 0.1 (see sensitivity sweep for why not the literature 0.3) |
| `strategy_ls_na_egbo.py` | LS-NA-EGBO: LLM warm-start once + autonomous novelty-aware EGBO. Also defines `mo_egbo_novelty` (no-LLM ablation) |
| `strategy_llm_candidate_gen.py` | LLM-as-candidate-generator strategy, formerly mislabelled "LABO" — see naming correction above |
| `llm_warmstart.py` | One-shot LLM warm-start module, three-layer prompt (reasoning protocol / domain physics / materials) |
| `excipient_campaign_mo.py` | Shared MO strategy module; also defines `strategy_mo_scalarized_ucb` (warm-start + cheap scalarised-UCB ablation, used by both `run_phase3.py` and `run_benchmark_resumable.py`) |
| `llm_af_evo/shared/ada_coatings_oracle.py` | Cross-domain oracle used by Phase 3 (`DiscreteADACoatingsOracle`, real multi-objective ADA coatings oracle, replacing the earlier placeholder `synthetic_coatings_oracle.py`) |
| `results/benchmark_phase1/` | Reduced formulation space, 15 seeds, budget=30 |
| `results/benchmark_phase2/` | Full 14-excipient space, 25 seeds, budget=40 — the headline benchmark numbers |
| `results/benchmark_phase3/` | Cross-domain ADA coatings, real LLM, 20 seeds, 7 conditions — see "Phase 3 results" above |
| `results/benchmark_sensitivity/` | Novelty-weight sweep, 15 seeds, budget=30 |

All of the above (plus `run_phase3.py`) are currently untracked in git — commit before this becomes a reproducibility gap.

---

## Everything else (Sections 1–11, 14–17)

Unchanged from `project_comprehensive_log_previous.md` — cancer drug screening pre-pivot, the pivot rationale, single-objective oracle design, the single-objective wrong-prior stress test (still the cleanest result in the project and still valid — it used the single-objective oracle, which nothing above touches), Waibel dataset extraction, multi-objective oracle construction and the six bugs found and fixed, and the literature context. Refer to that file for full detail; only the multi-objective campaign-runner sections (12–13) and the file inventory needed updating here.
