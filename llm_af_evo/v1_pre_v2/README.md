# v1_pre_v2 — the original build, before version numbering started

Everything from before this project adopted the `vN` naming convention
(v2 onward). `src/evolve_af.py`'s docstring calls this "L1 build, step 4":
FunBO-style programs-database evolution of `score_pool` AFs
(`af_interface.py`'s contract), fit against a fixed training set.
`src/evolve_af_2b.py` is the fix for that original design's off-policy
fitness gap (`fitness_common.py`'s docstring: a candidate's per-step
win/loss was scored against EGBO-novelty's own trajectory, never the
candidate's own hypothetical one) — `evolve_af_v2.py` onward (see
`../v2/README.md`) build on `evolve_af_2b.py`'s fitness approach, not the
original `evolve_af.py`'s.

## What's here

- `src/` — `evolve_af.py` / `evolve_af_2b.py` (the two evolution loops
  above), `af_interface.py` (7 hand-written seed AFs — the design later
  versions moved away from), `fitness_common.py`, `synthetic_mo_oracle.py`,
  `tunable_synthetic_oracle.py` (later moved to `../shared/`, see
  `../v3/README.md`), `da_coreg.py`, `compose_strategies.py` /
  `compose_interface.py` / `baseline_decomposition_strategies.py` /
  `composition_pilot_common.py` (candidate-composition strategy
  experiments).
- `experiments/` — a large collection of one-off pilot/diagnostic/sweep
  scripts (`run_*.py`, `diagnose_*.py`, `sweep_*.py`, `track_*.py`,
  `validate_*.py`) and their raw outputs (`*_results.json`,
  `*_raw.parquet`, `*_debug.csv`), plus `pareto_front_explorer.html` (a
  standalone viz page — see `campaign_exports/README.md`).
- `campaign_exports/` — has its own README; JSON dumps for
  `pareto_front_explorer.html`, not committed by default.

No consolidated results summary exists for this directory in this
document — the individual `*_results.json` files and the top-level
`project_comprehensive_log.md`/`llm_evolved_afs_comprehensive_log.md` are
the source of truth for what each experiment found.
