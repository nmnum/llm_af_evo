# Map: llm_af_evo/ thesis structure

## Destination

A clean, navigable directory structure for `llm_af_evo/` (the LLM-evolved
acquisition-functions track) that you can write the thesis from: current
pipeline code, one-off experiment/diagnostic scripts, dead ends, large
run/log data, and the existing findings docs each land somewhere legible,
with a clear line between "what shipped" and "what was tried and abandoned."
Root-level `ls_na_egbo/` (the excipient/EGBO benchmark) is out of scope for
this effort — its README already documents its structure.

## Notes

- Domain: multi-objective Bayesian optimization (BoTorch/pymoo) for
  pharma formulation (excipients, coatings); `llm_af_evo/` evolves
  acquisition functions via an LLM-driven, FunBO-style programs-database
  search.
- Primary source of truth for the project's history: `SESSION_LOG.md`
  (pre-v2 era: fitness 2a→2b→margin, the 8 mechanism gates in
  `L_COATINGS_FINDINGS.md`, compose_batch/DA-COREG dead ends) and
  `llm_evolved_afs_comprehensive_log.md` (v2 era: Jul 28–31 runs 1–4,
  hint-bug postmortems, mAb noise diagnostic, 3-seed re-validation).
  Both were reconstructed from code/results, not chat history — read
  them before guessing at what any script does.
- ~65MB of run data lives in `evolution_runs/`, `logs/`, `training_logs*/`.
- No git history exists yet for this directory (repo was just `git init`'d
  as part of this effort) — nothing here is a "recent change," treat file
  mtimes (Jul 20–31) as the only chronology available.
- Use `/grilling` for each ticket unless noted otherwise.

## Decisions so far

(none yet)

## Not yet specified

- How the reorganization actually gets executed (file moves/renames) once
  the structural decisions below are made — this map produces the plan,
  not the migration itself; whether that migration becomes its own
  follow-up pass or a "task" ticket appended here is still open.
- Whether/how the root-level `ls_na_egbo/` benchmark eventually gets a
  matching pass — explicitly deferred, not decided against.

## Out of scope

- Reorganizing root-level `ls_na_egbo/` (excipient/EGBO benchmark) files —
  already has a documented structure in its `README.md`; user chose to
  scope this effort to `llm_af_evo/` only.
