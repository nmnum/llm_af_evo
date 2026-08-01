Type: grilling
Status: resolved

## Question

What happens to the ~65MB of run data (`evolution_runs/`, `logs/`,
`training_logs*/`) and the per-experiment result JSONs at the top level
(`*_results.json`, `run_a.json`, `run_b.json`, etc.)? Options include:
keep everything in-repo, move the bulk data out of git (`.gitignore` +
external storage) while keeping small summary JSONs, or prune runs that
were superseded/invalidated (e.g. anything the hint-bug postmortems in
`llm_evolved_afs_comprehensive_log.md` flagged as bad). Should land on a
retention rule, not a file-by-file audit.

## Answer

Keep everything in git, in the shared `data/` bucket, no pruning:

- `evolution_runs/` (24MB), `logs/` (4.8MB), and all four
  `training_logs*/` directories (~35MB combined) move into `data/` as-is,
  committed.
- All top-level `*_results.json` files (the per-experiment summary JSONs
  already cited by name in `L_COATINGS_FINDINGS.md`'s results table, e.g.
  `composition_pilot_results.json`, `da_coreg_dtlz2_results.json`) also move
  into `data/`, not paired alongside the script that produced them —
  matches ticket 01's decision that `data/` is a single shared top-level
  bucket rather than split by era.

Rationale: ~65MB total is well within normal git repo size, and keeping
raw run data in-tree means every result cited in the thesis stays directly
reproducible/inspectable from a commit — no external storage or `.gitignore`
split to keep in sync. No pruning of runs flagged as undocumented (e.g.
`run_v2_mAb_gamma001_fixed`, per SESSION_LOG.md open question #5) — those
stay too; "no written analysis" is a docs gap (see ticket 04), not a reason
to discard the run data itself.
