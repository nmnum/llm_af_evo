Type: grilling
Status: open

## Question

What happens to the ~65MB of run data (`evolution_runs/`, `logs/`,
`training_logs*/`) and the per-experiment result JSONs at the top level
(`*_results.json`, `run_a.json`, `run_b.json`, etc.)? Options include:
keep everything in-repo, move the bulk data out of git (`.gitignore` +
external storage) while keeping small summary JSONs, or prune runs that
were superseded/invalidated (e.g. anything the hint-bug postmortems in
`llm_evolved_afs_comprehensive_log.md` flagged as bad). Should land on a
retention rule, not a file-by-file audit.
