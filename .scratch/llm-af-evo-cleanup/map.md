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

- [Directory structure](.scratch/llm-af-evo-cleanup/issues/01-directory-structure.md) — era-first split (`v1_pre_v2/`, `v2/`, each with `src/`+`experiments/`), plus shared top-level `data/` and `docs/`; no separate `archive/` bucket
- [Classify scripts](.scratch/llm-af-evo-cleanup/issues/02-classify-scripts.md) — all 49 top-level `.py` files sorted into `shared/`, `v1_pre_v2/{src,experiments}`, `v2/{src,experiments}` per the buckets above, derived from `SESSION_LOG.md`
- [Data retention](.scratch/llm-af-evo-cleanup/issues/03-data-retention.md) — keep all ~65MB of run data (evolution_runs/, logs/, training_logs*/) and all result JSONs in git, in shared `data/`; no pruning
- [Docs consolidation](.scratch/llm-af-evo-cleanup/issues/04-docs-consolidation.md) — all 5 findings/history docs stay separate in `docs/`, plus a new `docs/README.md` index; two known analysis gaps (run_v2_mAb_gamma001_fixed, Jul 31 diagnostics) noted but deferred, not resolved here

## Not yet specified

- How the reorganization actually gets executed (file moves/renames) now
  that the structural decisions (tickets 01-04) are all made — this map
  produced the plan, not the migration itself; whether that migration
  becomes its own follow-up pass or a "task" ticket appended here is still
  open. All four decision tickets are now resolved, so this is the last
  thing standing between the map and the destination.
- Whether/how the root-level `ls_na_egbo/` benchmark eventually gets a
  matching pass — explicitly deferred, not decided against.
- Writing up the two documented analysis gaps flagged in ticket 04
  (`run_v2_mAb_gamma001_fixed` has no analysis; the Jul 31 mu/sigma-dominance
  and win-overlap diagnostics never reached a conclusion) — explicitly ruled
  out of *this* map's scope (it's research work, not structure), but noted
  here as a known follow-up.

## Out of scope

- Reorganizing root-level `ls_na_egbo/` (excipient/EGBO benchmark) files —
  already has a documented structure in its `README.md`; user chose to
  scope this effort to `llm_af_evo/` only.
