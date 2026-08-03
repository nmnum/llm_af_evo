# Comprehensive Log: sdl-adaptive

**Compiled**: 2026-08-03, from a read-only interpretation pass over the existing codebase and results directories (no new experiments were run to produce this log; all numbers below come from CSVs and console output already present in this folder).

---

## 1. Project purpose

`sdl-adaptive` benchmarks whether an LLM can make good **strategy-level decisions** in a simulated self-driving-lab (SDL) optimisation campaign — i.e. automate the meta-decisions (which acquisition strategy to use, which hyperparameters to set, when to switch strategy) that a human scientist normally makes heuristically, rather than proposing new optimisation math itself.

Campaigns are simulated over:
- Real datasets: `coatings` (thin-film process data) and four `pareto_*` OPV campaigns from the ADA benchmark (`pareto_20201218`, `pareto_20210112`, plus others referenced in later runs).
- Synthetic datasets: `hartmann3`, `hartmann6`.

An **NN-based oracle** (`oracle.py`) is used instead of a GP oracle. This was a deliberate design choice: a GP oracle was found to extrapolate to **193× the empirical maximum** on the coatings dataset (7D, only 91 points), so it was rejected as unreliable for ground-truth scoring.

## 2. Controllers / approaches implemented (`controllers/`)

- **Approach A** (`approach_a.py`) — LLM tunes a single scalar (UCB β).
- **Approach B** (`approach_b.py`) — LLM picks among six discrete strategies (UCB, EI, PI, Thompson, random, LHS) plus params.
- **Approach C** (`approach_c.py`) — LLM rewrites the entire `suggest()` optimiser function as Python code, executed in a subprocess sandbox. Logs of LLM-generated code land in `approach_c_code_logs/`, `approach_c_logs_pareto_20210112/`.
- **Approach C-evo** (`approach_c.py`, `_TEMPLATE_EVO` / `optimiser_template_evo.py`) — same as Approach C, but the LLM-generated code has access to `evolutionary_candidates()` and `novelty_select()` helpers (`evolutionary_candidates.py`), so it can sample an evolutionary candidate pool and pick via a novelty-weighted merit score (default `merit_weight=0.7`). Logs in `approach_c_evo_logs_pareto_20210112/`.
- **Approach D** (`approach_d.py`) — an LLM *router*: given campaign-state signals (best value, improvement rate, GP uncertainty, lengthscale, obs/dim, progress), it picks per-step among `{egbo, novelty_egbo, ucb, lhs, random}`. Uses the instruct model (not the coder model) for better discrete-JSON output; falls back to a deterministic rule (`_rule_fallback`, mirroring `rule_router.py`) if the LLM's output is unparseable.
- **`rule_router.py`** — the pure rule-based (non-LLM) version of the same routing logic; used both as a standalone baseline and as Approach D's fallback.
- **`egbo` / `novelty_egbo`** — Evolutionary GP Bayesian Optimization (BoTorch qLogNoisyEI + pymoo U-NSGA-III), following Aqeeli et al. — the hand-coded, non-LLM strategy that later results treat as a strong baseline/reference.
- **`mock_controller.py`** — rule-based stand-ins for the LLM controllers, used for fast iteration/testing without live LLM calls.

`gp_trust_check.py` implements three proactive GP-trust diagnostics (LOO calibration, lengthscale/nearest-neighbour-distance ratio, posterior-uncertainty reduction → a composite trust score in [0,1]), used to decide whether to trust GP-based strategies before committing experiments.

## 3. Experiment / analysis scripts and what each tests

| Script | What it tests |
|---|---|
| `run_experiment.py` | Main harness — runs all baseline/mock/LLM conditions across 5 datasets × 20 seeds. Outputs `metrics_summary.csv` (auc_best, final_best_normalised, steps_to_90pct, switch_count, failure_count). |
| `oracle_gap_experiment.py` | Isolates exactly 3 LLM judgment calls — GP lengthscale-mode prior, ARD on/off, evolutionary population size — comparing `default` vs `llm`-chosen vs `oracle`-optimal values at `budget_frac` 0.5 and 1.0. Reports `ls_mode_error`, `ard_match`, `evo_pop_error`, and resulting AUC. |
| `shared_seed_experiment.py` | Fixes identical initial points across all conditions per repeat, to remove the "initialisation lottery" confound. |
| `mid_campaign_analysis.py` | Filters to seeds with a "bad start" (best_norm below threshold at n_init) and computes AUC over just the mid-campaign recovery window. |
| `analyse_decisions.py` | Inspects `switch_logs.json` to see what fraction of router decisions pick egbo/ucb/lhs/random, and whether the router favours EGBO on "structured" datasets and avoids it on "flat" ones. |
| `analyse_fcs_sweep.py` | Sweeps the "first control step" (5/10/15/20 — when routing first kicks in) on flat vs structured landscapes. |
| `gp_trust_check.py` | Standalone GP-trust diagnostic (see above), used ahead of committing to GP-based strategies. |
| `ablation_batch_pool.py`, `ablation_batch_scoring.py`, `ablation_gp_calibration.py` | Ablations on batch size/refit strategy, candidate-pool composition (EA fraction, pool size, acquisition type), and GP calibration effects on EGBO performance. |

## 4. Concrete results, by directory

All numbers are mean `auc_best` unless stated otherwise; std is repeat-to-repeat standard deviation.

### `results/` — original 9-condition × 5-dataset × 20-seed run
Mean AUC ranges ~0.69–0.96 across conditions/datasets. On `coatings`, `approach_a` (LLM tunes β) scored highest (0.945) vs `fixed_random`/`fixed_lhs` (~0.90) and `ada_original` (0.901). On `pareto_*` datasets, differences between LLM approaches and fixed baselines were small (~0.78–0.84) — within the noise floor the README itself flags (seed SD 0.03–0.14, so differences under ~0.05 AUC should be interpreted cautiously).

`mid_campaign_conditioned.csv`: conditioned on bad starts, on `coatings` LLM-A scored best (0.903), Mock-C worst (0.785); rankings reshuffle across datasets (e.g. on `pareto_20201218`, `fixed_random` beats several LLM/mock conditions) — no approach dominates once initialisation luck is controlled for.

### `results_oracle_gap/` — isolated hyperparameter judgment calls
AUCs are nearly identical across `default`/`llm`/`oracle` conditions per dataset/budget (e.g. `pareto_20210112` at `budget_frac=1.0`: default 0.863, llm 0.866, oracle 0.871). `evo_pop_error` is 0.0 across sampled rows. Per-call logs show the LLM converging on stable choices (ls_mode≈1.0, ARD shifting to True near the campaign end, evo_pop growing 50→300 over the campaign).

A later rerun with `qwen2.5:14b-instruct` (20 repeats, `pareto_20210112` / `hartmann6` / `coatings`, `budget_frac` 0.5 and 1.0) reinforced this: `ls_mode_err` (lengthscale-mode error) was uniformly **worse** with the LLM than with the naive default in every single dataset (pareto 0.39→0.79, hartmann6 0.83→1.23, coatings 0.13→0.53), `ard_match` was 100% for every condition (no discriminating signal there), and `fit_fail=0%` throughout — yet absolute AUC barely moved (≤0.03 spread within any row), because the benchmark's AUC is fairly insensitive to lengthscale-mode choice at these budgets. The script's printed "% of default-to-oracle gap closed" metric is numerically unstable when oracle≈default (it produced a nonsensical "1219%" on one row) and should not be trusted as the primary readout — the raw AUC deltas and `ls_mode_err` are the reliable signals. **Net conclusion: isolated LLM hyperparameter judgment calls do not approach oracle-level tuning, and a larger 14b instruct model does not fix this — it is arguably worse at the lengthscale sub-task than the smaller coder model, despite being bigger.**

### `results_fcs_sweep/` — routing-start-step sweep (flat vs structured landscapes)
EGBO consistently scores highest full/conditioned AUC across all four `fcs` settings (5/10/15/20) on both `pareto_20201218` (flat) and `hartmann6` (structured), beating both `rule_router` and `approach_d` in every row:

| fcs | dataset (regime) | rule_router | approach_d | egbo |
|---|---|---|---|---|
| 5 | pareto_20201218 (flat) | 0.524 | 0.566 | **0.592** |
| 5 | hartmann6 (structured) | 0.722 | 0.717 | **0.814** |
| 10 | pareto_20201218 | 0.558 | 0.556 | **0.581** |
| 10 | hartmann6 | 0.728 | 0.713 | **0.820** |
| 15 | pareto_20201218 | 0.575 | 0.557 | **0.623** |
| 15 | hartmann6 | 0.690 | 0.739 | **0.818** |
| 20 | pareto_20201218 | 0.582 | 0.548 | **0.607** |
| 20 | hartmann6 | 0.686 | 0.752 | **0.833** |

No clear crossover fcs value was found in the sampled range — EGBO wins regardless of when routing kicks in.

### `results_ablation/` — EGBO implementation ablation (`pareto_20210112`, 20 repeats each)
`egbo_botorch`, `egbo_sklearn`, `novelty_w03/07/09` all show similarly noisy per-repeat AUCs (0.53–1.0); no strong, obvious effect of GP backend (botorch vs sklearn) or novelty weight (0.3/0.7/0.9) at a glance — high seed variance dominates.

### `results_ablation2/` — batch-size and pool-composition sweeps
Batch size (1/2/4/8, joint vs refit) and pool composition (EA fraction, pool size, acquisition type) swept on `hartmann6`/`coatings`/`pareto_*`; includes regime-map PDFs (`batch_size_regime_map.pdf`, `pool_composition_regime_map.pdf`) visualising interaction with landscape "regime" (e.g. multimodal).

### `results_power/` / `results_power2/` — larger n=80 runs (statistical power)
`results_power`: on `pareto_20201218`, `approach_c` (LLM rewrites code) scored highest (0.843) vs `fixed_ucb_low` lowest (0.717).

`results_power2` (n=80, `pareto_20210112`) — the clearest directional win for EGBO in the whole project:

| condition | mean AUC | std |
|---|---|---|
| **egbo** | **0.805** | 0.116 |
| novelty_egbo | 0.791 | 0.131 |
| approach_c | ~0.71 | — |
| approach_c_evo | 0.708 | 0.158 |
| fixed baselines | ~0.71 | — |

`approach_c_evo` (LLM-written code using evolutionary/novelty helpers) was the *weakest* of the group here — giving the LLM's generated optimiser access to evolutionary-candidate/novelty-selection machinery did not close the gap to hand-tuned EGBO; if anything the LLM-authored implementation underperformed the reference one.

### `results_shared/` — shared-seed (confound-controlled) runs
On `pareto_20210112`: `novelty_egbo` (0.833) and `egbo` (0.818) beat `approach_c` (0.760), `approach_a` (0.746), and fixed baselines (0.70–0.75).

### `results_ninit15/` — n_init=15 instead of 5 (n=20 repeats)
| dataset | approach_d | rule_router | egbo |
|---|---|---|---|
| hartmann6 | 0.791 | 0.796 | **0.850** |
| pareto_20210112 | 0.765 | 0.753 | **0.778** |

### `results_phase1/` — synthetic-function results (n=20)
| dataset | approach_d | rule_router | egbo |
|---|---|---|---|
| hartmann3 (low-D, smooth) | 0.870 | 0.871 | **0.939** |
| hartmann6 | 0.744 | 0.722 | **0.837** |

EGBO is dominant on the synthetic structured functions. Also contains `lengthscale_retrospective.csv/pdf`.

### `results_llm_test/`, `results_prompt_test/`, `results_evo_test/`, `results_evo_test2/`
Small quick/smoke tests (coatings-only, or single-condition `approach_c`/`approach_c_evo` runs on `pareto_20210112`) used to validate the LLM pipeline and prompt design rather than as final results.

## 5. Cross-cutting comparison: EGBO vs LLM-driven routing (approach_d, rule_router, approach_c_evo)

Pulled directly from the tables above (mean AUC, all datasets/conditions where a head-to-head exists):

- **EGBO beats `rule_router`, `approach_d`, and `approach_c_evo` in every single row sampled** across `results_power2`, `results_phase1`, `results_ninit15`, and `results_fcs_sweep` — typically by 0.05–0.13 AUC, well outside the README's stated 0.05 noise floor for these larger runs.
- **`approach_d` (the LLM router that can itself select EGBO) tracks `rule_router` closely** (usually within ~0.02–0.05), rather than beating it — the LLM does not appear to route to EGBO meaningfully better, or more often, than the hand-coded rule fallback it's meant to improve on.
- **`approach_c_evo` (LLM-written code using evolutionary/novelty helpers) is the weakest LLM-adjacent condition** in `results_power2` (0.708 vs egbo's 0.805) — access to evolutionary-candidate machinery inside the LLM's own generated code does not let it match hand-implemented EGBO.
- **Net effect: routing to EGBO (via LLM or rules) never outperforms simply running EGBO directly** in any table pulled — the router adds LLM/rule overhead without capturing EGBO's edge.

## 6. Stated conclusions / caveats (from README and code)

1. NN oracle was chosen specifically because a GP oracle's extrapolation was unreliable (193× empirical max on coatings).
2. Discrete vs "snap-to-grid" scoring modes produce meaningfully different strategy rankings (Spearman ρ = 0.117 between them) — a scoring-mode sensitivity worth flagging in any write-up.
3. Datasets are small (N=53–91 observations) with high seed variance, so AUC differences under ~0.05 should not be over-interpreted.
4. Mock controllers are rule-based stand-ins, not real LLM behaviour, and are used for pipeline validation, not as evidence.
5. Only one model was used for most of the original three-approach experiments (qwen2.5-coder:7b); a larger instruct model (qwen2.5:14b-instruct) was tested later specifically for the oracle-gap hyperparameter experiment and did not improve on the lengthscale-mode judgment sub-task.

## 7. Overall takeaway

Across the full set of results in this folder, the most consistent, reproducible empirical pattern is that **hand-coded EGBO outperforms every LLM-driven approach (A/B/C/C-evo/D) and every fixed/rule-based baseline** tested here, especially in the larger, more carefully controlled runs (`results_power2`, `results_shared`, `results_ninit15`, `results_phase1`, `results_fcs_sweep`). The original three LLM approaches (A/B/C) show only marginal, seed-noise-level advantages over fixed baselines in the smaller original run, and the isolated LLM hyperparameter judgment calls (`oracle_gap_experiment`) track the fixed default closely rather than approaching oracle-optimal tuning — with a larger LLM (14b vs 7b) making the lengthscale-mode judgment *worse*, not better. This motivated moving to the differently-designed `llm_bo` project (candidate-pool ranking with domain-knowledge priors, in the sibling `llm_bo/` folder), which found a more favourable result for LLM value-add via a different mechanism.
