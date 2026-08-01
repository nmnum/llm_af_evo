Type: grilling
Status: resolved
Blocked by: 01

## Question

Of the ~50 scripts in `llm_af_evo/`, which are the current/active pipeline
(e.g. `evolve_af_v2.py` and whatever it depends on), which are one-off
pilots/diagnostics worth keeping as citable evidence (the `run_*_pilot.py`,
`diagnose_*`, `validate_*`, `check_*` scripts), and which are superseded
dead ends to archive (e.g. `af_interface.py` vs `af_interface_v2.py`,
`evolve_af.py` vs `evolve_af_2b.py` vs `evolve_af_v2.py`, the
`compose_*`/`da_coreg*` dead ends SESSION_LOG.md already calls out)? Produces
a per-file mapping into the buckets fixed by ticket 01.

## Answer

Classified all 49 top-level `.py` files by reading `SESSION_LOG.md` in full
(the independently-verified reconstruction of the project's history) plus
inspecting `sandbox.py` and `test_af_generalization.py` directly, since
SESSION_LOG.md didn't pin their era unambiguously. Rule used: "src" =
reusable implementation modules (interfaces, strategy/oracle libraries,
infra); "experiments" = one-off runner/debug/diagnose/validate/sandbox
scripts that produced a specific citable result; "shared" = infra actually
reused across both v1 and v2 (confirmed via SESSION_LOG.md's explicit
cross-era mentions, not guessed).

### `shared/`
- `sandbox.py` — subprocess execution sandbox for evolved AF code; era-agnostic execution infra, not tied to v1 or v2's interface
- `full_replay.py` — the "2b" on-policy fitness fix; v2 kept full-replay, only changed the aggregation metric (win-rate → margin)
- `mock_mutator.py` — deliberately-weak sanity-floor mutator used across all evolution runs
- `ada_coatings_oracle.py` — coatings domain oracle, used by both pre-v2 coatings pilots and the v2 coatings run
- `generate_training_set.py`, `generate_coatings_training_set.py` — per SESSION_LOG.md §2, the 100-campaign v2 training sets were (re)generated with these same scripts
- `make_figures.py` — generic figure-gen utility, not tied to one era's findings doc

### `v1_pre_v2/src/`
- `af_interface.py`, `evolve_af.py`, `evolve_af_2b.py`, `fitness_common.py` — v1 interface/engine/fitness ("2a"/"2b" generations)
- `compose_interface.py`, `compose_strategies.py`, `da_coreg.py` — compose_batch/DA-COREG dead-end implementation (Part 7-8 gates)
- `gen_interface.py` — geometry-only candidate-generation dead end (Part 4 gate)
- `baseline_decomposition_strategies.py`, `composition_pilot_common.py` — shared helper libs for the Part 3/mAb-pilot-gauntlet pilots
- `synthetic_mo_oracle.py` — ZDT1/DTLZ2 synthetic positive-control oracle used by the compose/DA-COREG gates

### `v1_pre_v2/experiments/`
- `run_mock_subset.py`, `diagnostic_true_oracle_hv.py`, `sweep_lambda.py` — Jul 20-21 bootstrap
- `validate_2b_seed.py`, `validate_l1.py`, `run_2b_diagnostic.py` — Jul 22-24 rank-collapse-bug diagnosis
- `run_composition_pilot.py`, `run_composition_pilot_coatings.py`, `run_generation_pilot.py`, `run_batch_size_ablation.py`, `run_mc_hvi_pilot.py`, `run_baseline_decomposition_pilot.py`, `run_unsga3_pool_pilot.py` — the mAb pilot gauntlet (Part 3-6 gates)
- `check_objective_correlation.py` — diagnosed the fake-3-objective coatings bug (Part 2)
- `run_coatings_replicates.py`, `run_coatings_generalization.py` — coatings domain replication (Part 2)
- `compose_sandbox.py`, `debug_compose.py`, `run_synthetic_compose_pilot.py`, `run_da_coreg_pilot.py`, `run_da_coreg_no_qnehvi_pilot.py`, `diagnose_da_coreg_fit.py` — compose_batch/DA-COREG gate runs and postmortem (Part 7-8)
- `gen_sandbox.py`, `debug_generation.py` — geometry-only generation gate runs (Part 4)
- `run_synthetic_generalization.py` — synthetic generalization check
- `test_af_generalization.py` — regression test for v1's objective-name parameterization fix (imports `af_interface`/`fitness_common`, both v1)

### `v2/src/`
- `af_interface_v2.py`, `evolve_af_v2.py` — v2 interface + engine (margin-based fitness, minimal-seed design)

### `v2/experiments/`
- `diagnose_mu_sigma_dominance.py`, `diagnose_win_overlap.py`, `mab_noise_diagnostic.py`, `validate_population_2b.py` — the Jul 31 anomaly-chasing session (§7), all written/updated same day against the v2 coatings run

All 49 files accounted for. Two files (`sandbox.py`, `test_af_generalization.py`)
weren't unambiguous from SESSION_LOG.md's narrative alone and were resolved by
reading their docstrings directly — flagging in case that reading is wrong
once the actual file moves happen.
