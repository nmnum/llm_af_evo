# shared — modules reused across version folders

Code and data that multiple `vN/` folders import directly rather than
each duplicating. Per this project's versioning convention (see
`af_interface_v3.py`'s docstring on why `STRATEGY_HINTS` is copied, not
shared, across releases), only genuinely version-independent
infrastructure lives here — evolution loops and seed/hint definitions
stay forked per version.

## What's here

- `full_replay.py` — `strategy_evolved_af`: runs a full sequential BO
  campaign against a real oracle with a candidate AF picking every batch
  (the "2b" fitness fix — see `v1_pre_v2/README.md`). Also
  `reconstruct_oracle`, with per-`oracle_family` branches
  (`excipient`/`coatings`/`mAb`/`tunable`/`dtlz2`/`zdt1`/...).
- `sandbox.py` — runs a candidate's `score_pool` code in a sandboxed
  subprocess against a built `context` dict; threads through optional
  extra context fields (`obj_correlation`, `front_boundary_std`,
  `pool_acq_value` → `acq_value_norm`, etc.) as they were added by later
  versions.
- `tunable_synthetic_oracle.py` — `TunableSyntheticMOOracle`, the
  controlled synthetic domain (moved here from `v1_pre_v2/src/` in v3 so
  `full_replay.py` could import it without reaching into `v1_pre_v2`).
- `ada_coatings_oracle.py` — the real coatings-domain oracle.
- `mock_mutator.py` — mock-mode AF-code mutation, used by non-real-LLM
  smoke tests (e.g. `v2/experiments/evolution_runs/run_v2_mAb_dro_hvi`).
- `generate_training_set.py`, `generate_coatings_training_set.py` — fixed
  training/held-out campaign-set generators for the excipient and
  coatings domains (the tunable/dtlz2/zdt1/noisy-synthetic equivalents
  live per-version, e.g. `v3/experiments/generate_tunable_training_set.py`,
  `v5/experiments/generate_dtlz2_training_set.py`).
- `generate_coreg_steps.py`, `diagnose_da_coreg_posterior.py` — DA-coreg
  (data-augmented co-regionalization) generation/diagnostic scripts.
- `make_figures.py` — shared figure-generation helper.
- `coatings_data/` — raw coatings campaign CSVs backing
  `ada_coatings_oracle.py`.
