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

## 7. 2026-08-09/10 session: `results_combined` (5-condition-group merge), EGBO silent-fallback bug fix, mid-campaign analysis, and a data-loss incident

This session merged `results_baselines`, `results_a`, `results_b`, `results_c`, `results_d`/`results_d_fixed`, and `results_egbo` into a single `results_combined/metrics_summary.csv` (14 conditions × 5 datasets × 20 seeds = 1400–1420 rows depending on the pass) and regenerated all `figures_combined/*.png` from it, once `approach_c` finished its last two pending Pareto datasets. **As of writing this section, the raw per-condition output directories and `figures_combined/` that produced the numbers below were deleted from the shared checkout by an unknown process (not by this session) before they were committed to git — see §7.5. The numbers/conclusions here remain valid (they were computed and recorded before the deletion) but the underlying `curves.npy`/`switch_logs.json` artifacts need to be regenerated to be re-verified or re-plotted. §7.6 gives the exact re-run commands.**

### 7.1 Bug found and fixed: silent EGBO→random fallback in `strategies.py`

`strategies.py`'s `egbo()` / `novelty_egbo()` / `egbo_discrete()` / `novelty_egbo_discrete()` — the functions an LLM controller (`approach_d`) calls when it chooses strategy name `"egbo"`/`"novelty_egbo"` mid-campaign — silently substituted `random_search`/`random_discrete` whenever `torch`/`botorch`/`gpytorch`/`pymoo` weren't importable in the running interpreter, with only a one-time log warning and **no entry in `switch_logs.json`'s `failures` list**. Confirmed via `readlink -f /proc/<PID>/exe` that the historical `approach_d` run (and other campaigns) actually executed under the bare `/usr/bin/python3.12` interpreter, which lacks these dependencies (`import torch` → `ModuleNotFoundError`), not the botorch-enabled `~/sdl_protein/sdl_prot/bin/python3` env.

Per-dataset "egbo" request share in the old `approach_d` `switch_logs.json` (i.e. how often the LLM asked for egbo and silently got random instead): coatings ~10% (35/360), pareto_20201218 ~50% (119/240), pareto_20201223 ~46% (111/240), pareto_20210104 ~45% (90/200), pareto_20210112 ~40% (112/280). This means up to ~50% of `approach_d`'s "egbo" decisions on Pareto datasets were confounded with random search in the pre-fix data.

**Fix**: the four functions now `raise RuntimeError` instead of silently falling back, so `simulator.py`'s existing per-step `except Exception: failures.append(step)` handler catches it, falls back to random itself, *and* records the step in `switch_logs.json.failures` — the same failure mode is now visible/counted instead of invisible. `simulator.py`'s stale comment describing `"egbo"` as an unmapped strategy name was also corrected (it *is* in `STRATEGY_MAP`; the fallback happens via the dependency check in `_egbo_module()`, not the unmapped-strategy branch).

Re-running `approach_d` under the correct interpreter (`results_d_fixed/`, same 20 seeds via the shared `generate_init_points(oracle, n_init=5, seed=seed, ...)` mechanism, so directly comparable to the old run) shifted `auc_best` mostly on the highest-bug-rate dataset: `pareto_20210104` 0.732 → 0.776 (the dataset with ~45% egbo-call failure rate pre-fix); `coatings` 0.923 → 0.937; `pareto_20201218`/`pareto_20201223` ~flat; `pareto_20210112` 0.798 → 0.789 (slightly down). `failure_count` in `metrics_summary.csv` read 0 for both old and new runs in this pass, which is a metric-plumbing gap worth checking separately — `switch_logs.json.failures` (not currently surfaced into `metrics_summary.csv`) is the more reliable signal for this bug.

### 7.2 What each LLM approach actually does (from `switch_logs.json` decision logs, aggregated across all seeds/datasets)

- **LLM-A**: 100% `ucb` (1320/1320 decisions) — never switches strategy type, only tunes β (mean≈220.8, range 12.5–1000).
- **LLM-B**: `lhs` 45% (588/1320), `ei` 36% (473/1320), `ucb` 19% (259/1320, β mean≈12.8) — mean 2.18 type-switches per seed-run.
- **LLM-C**: `_custom` 1039/1040 decisions (re-generates its own `suggest_point()` code every controller-interval step), `random` only once, 2 logged failures across 80 seed-runs — essentially never reverts to the acquisition-function menu.
- **LLM-D**: `lhs` 193, `ucb` 660 (β mean≈12.3), `egbo` 467 out of 1320 total — mean 3.96 switches/run, the highest switch rate of the four, and the only approach that invokes EGBO at all.

### 7.3 Full-campaign AUC interpretation (`results_combined`, `auc_best`, full budget)

- **EGBO dominates**: statistically significant advantage over standard baselines (random/LHS/UCB-low/UCB-high/EI) on 4/5 datasets (paired t-test on shared seeds); `pareto_20201218` is a genuine, reproducible exception where the gap narrows.
- **LLM approaches vs baselines**: largely statistically indistinguishable from simple fixed baselines. The one consistently significant win across all four LLM approaches is vs. `fixed_ucb_low` (β=0.2, the weakest baseline) — p<0.01 in every case. Beyond that, LLM-A beats `fixed_ei` (p=0.044); none of the four beats `fixed_random`, `fixed_lhs`, or `fixed_ucb_high` significantly.
- **Ranking**: LLM-A is the strongest LLM approach and closest to competitive with strong fixed baselines (its 100%-UCB behavior effectively makes it a self-tuning UCB baseline). LLM-C is mid-pack. LLM-D is the most dataset-dependent and, pre-fix, was confounded by the EGBO bug on Pareto sets.
- **EGBO vs LLM approaches directly**: EGBO beats all four LLM approaches, significantly, essentially everywhere it's compared.

### 7.4 Mid-campaign (`h=40` of ~53–91-step budget) analysis — `figure2_auc_bars_h40.png` / `figure2_final_at_h40.png`

Mean AUC@40 across datasets: EGBO 0.845, UCB β=400 0.790, LLM-A 0.788, LLM-C 0.785, LHS 0.781, LLM-B 0.777, Random 0.775, LLM-D 0.771, EI 0.762, UCB β=0.2 0.732.

- **EGBO's edge is front-loaded** — already present and significant (p<0.0001 vs all four LLM approaches) by step 40, not something accrued only late in the campaign.
- Rankings at h=40 closely track the full-budget rankings — no LLM approach shows a "slow starter, catches up later" pattern.
- LLM-A remains closest to competitive (0.788 vs UCB-high's 0.790, not distinguishable, p=0.83).
- LLM-D is weakest mid-campaign (0.771, at/below random), even post-bugfix — its mixed lhs/ucb/egbo switching isn't clearly paying off relative to just picking one baseline strategy and sticking with it.

### 7.5 Statistical power check: is n=20 seeds enough?

Computed Cohen's d and required-n-for-80%-power for every LLM-vs-baseline pairing on `auc_best` (pooled n=100, 5 datasets × 20 seeds). Conclusion: **no, n=20 does not need to be increased broadly.** Near-zero effects (d≈0.01–0.09, e.g. `approach_a` vs `fixed_random`) would need 800–100,000+ seeds to resolve — not a power problem, the true effect is ~0. Comparisons that already look meaningful (LLM vs UCB-low, d≈0.28–0.41; LLM-A/C vs EI, d≈0.20–0.26) are already significant at n=20. The one genuinely borderline case is `approach_d` vs `fixed_lhs` (p=0.049, d=-0.20, n_needed≈198) — worth a targeted larger run *only* on that specific pair if it matters for the thesis narrative, not a blanket 80-seed re-run of everything.

### 7.6 Data-loss incident and recovery commands

**Resolved (confirmed 2026-08-11) — recovery was already done, just sitting on an unmerged branch.** This section originally read as still-open (see the re-run commands below) because it was written without visibility into a parallel worktree (`sdl-adaptive-improvements`) that had already run this exact recovery the night before this entry was committed — `results_a/b/c/d_fixed/baselines/egbo`, `figures_combined/`, and `results_combined/metrics_summary.csv` were regenerated and committed there (commits `2bcad66`/`ff47199`/`8b31679`, 2026-08-09 21:26–21:30 UTC), roughly 10 hours before this doc entry was written (2026-08-10 07:43 UTC) describing them as lost. Merged into this branch on 2026-08-11 — the data described as missing below is now present in the checkout; the re-run commands are kept for reference (they remain the correct recovery procedure if this ever happens again), not because they still need running.

After this session's `results_a/b/c/d_fixed/baselines/egbo` and `figures_combined/` were generated (feeding §7.1–7.4 above) but before they were committed to git, they were found deleted from the shared checkout (`/home/nehamungale/ls_na_egbo/sdl_adaptive/`) by a process not run by this session — plausibly a concurrent session sharing the same checkout. `git fsck --lost-found --full` found two dangling `git stash` WIP commits from around the same timestamp, but plain `git stash` only captures *tracked*-file changes; these untracked result directories were never stashed and are not recoverable from git objects. The `COMMIT_EDITMSG` file (which persists across attempted-but-not-completed commits) misleadingly suggested a commit had happened when it hadn't — `git log`/`git show` are the only reliable way to confirm a commit landed, not `COMMIT_EDITMSG`.

**Not lost**: the code (`strategies.py`/`simulator.py` fixes, `combine_results.py`, `figure2_short_horizon.py`, `figure2_final_full_budget.py`) and this log entry, since seeding is fully deterministic (`generate_init_points(oracle, n_init=5, seed=seed, ...)`, keyed only by `seed`) — a re-run reproduces identical initial conditions to what produced §7.1–7.4.

Re-run commands (from `sdl_adaptive/`, using `~/sdl_protein/sdl_prot/bin/python3` so EGBO's real dependencies are present):

```bash
# 1. Baselines (fast, no LLM)
PYTHONPATH=. ~/sdl_protein/sdl_prot/bin/python3 run_experiment.py --n_seeds 20 --out_dir results_baselines/ \
  --conditions fixed_random fixed_lhs fixed_ucb_low fixed_ucb_high fixed_ei ada_original \
               mock_approach_a mock_approach_b mock_approach_c

# 2. EGBO / novelty-EGBO (fast-ish, no LLM)
PYTHONPATH=. ~/sdl_protein/sdl_prot/bin/python3 run_experiment.py --n_seeds 20 --out_dir results_egbo/ \
  --conditions egbo novelty_egbo

# 3-5. LLM approaches (slow — real Ollama calls, qwen3-coder:30b)
PYTHONPATH=. ~/sdl_protein/sdl_prot/bin/python3 run_experiment.py --n_seeds 20 --out_dir results_a/ --conditions approach_a --model qwen3-coder:30b
PYTHONPATH=. ~/sdl_protein/sdl_prot/bin/python3 run_experiment.py --n_seeds 20 --out_dir results_b/ --conditions approach_b --model qwen3-coder:30b
PYTHONPATH=. ~/sdl_protein/sdl_prot/bin/python3 run_experiment.py --n_seeds 20 --out_dir results_c/ --conditions approach_c --model qwen3-coder:30b
PYTHONPATH=. ~/sdl_protein/sdl_prot/bin/python3 run_experiment.py --n_seeds 20 --out_dir results_d_fixed/ --conditions approach_d --model qwen3-coder:30b

# 6. Rebuild combined view + figures
PYTHONPATH=. ~/sdl_protein/sdl_prot/bin/python3 combine_results.py
PYTHONPATH=. ~/sdl_protein/sdl_prot/bin/python3 analyse.py --results_dir results_combined --out_dir figures_combined
PYTHONPATH=. ~/sdl_protein/sdl_prot/bin/python3 figure2_short_horizon.py
PYTHONPATH=. ~/sdl_protein/sdl_prot/bin/python3 figure2_final_full_budget.py

# 7. Commit immediately — do not let untracked LLM-run output sit uncommitted in the shared checkout again
git add results_baselines results_egbo results_a results_b results_c results_d_fixed \
        results_combined/metrics_summary.csv figures_combined/*.png \
        strategies.py simulator.py combine_results.py figure2_short_horizon.py figure2_final_full_budget.py
git commit -m "sdl_adaptive: re-run and document full results after prior uncommitted run was lost"
git push origin master
```

## 8. Overall takeaway

Across the full set of results in this folder, the most consistent, reproducible empirical pattern is that **hand-coded EGBO outperforms every LLM-driven approach (A/B/C/C-evo/D) and every fixed/rule-based baseline** tested here, especially in the larger, more carefully controlled runs (`results_power2`, `results_shared`, `results_ninit15`, `results_phase1`, `results_fcs_sweep`). The original three LLM approaches (A/B/C) show only marginal, seed-noise-level advantages over fixed baselines in the smaller original run, and the isolated LLM hyperparameter judgment calls (`oracle_gap_experiment`) track the fixed default closely rather than approaching oracle-optimal tuning — with a larger LLM (14b vs 7b) making the lengthscale-mode judgment *worse*, not better. This motivated moving to the differently-designed `llm_bo` project (candidate-pool ranking with domain-knowledge priors, in the sibling `llm_bo/` folder), which found a more favourable result for LLM value-add via a different mechanism.
