Type: grilling
Status: resolved

## Question

`llm_af_evo/` currently has four narrative/findings docs written at
different times for different purposes: `SESSION_LOG.md` (pre-v2 history),
`llm_evolved_afs_comprehensive_log.md` (v2 era, companion to the above),
`L_COATINGS_FINDINGS.md` (the 8 mechanism gates), and two dated research
notes (`research_fitness_alternatives.md`, `research_seed_population.md`).
For thesis writing, should these stay as separate standing documents (each
linked from a top-level index/README), get merged into one canonical
narrative, or get restructured some other way (e.g. one doc per thesis
chapter/section they'll feed)? Fixes how future findings docs get added
too.

## Answer

All five docs stay separate in `docs/` — each already has a distinct
purpose (history narrative vs. per-run numeric detail vs. central findings
vs. literature surveys) and merging or re-cutting them by thesis chapter
would be a bigger rewrite than this reorganization calls for. Add a new
`docs/README.md` index that states what each doc is and a suggested read
order, since right now that map only exists implicitly inside
SESSION_LOG.md's own prose cross-references (e.g. it names
`llm_evolved_afs_comprehensive_log.md` as its companion but nothing points
the other way, and neither mentions the two `research_*.md` surveys).

```
docs/
  README.md                              # index: what each doc is, read order
  SESSION_LOG.md                         # pre-v2 history narrative
  llm_evolved_afs_comprehensive_log.md   # v2 run-by-run detail (companion to SESSION_LOG)
  L_COATINGS_FINDINGS.md                 # central Parts 1-8 findings
  research_fitness_alternatives.md       # lit survey
  research_seed_population.md            # lit survey
```

**Explicitly out of scope for this effort**: the two documentation gaps
SESSION_LOG.md's own closing note flags — (1) `run_v2_mAb_gamma001_fixed`
(the last run, 22:09 Jul 31) has no written analysis, and (2) the Jul 31
mu/sigma-dominance and win-overlap diagnostics never reached a written
conclusion. `docs/README.md` should note both as known gaps (so they stay
visible) but resolving them is separate research work, not a structure
decision — tracked as a follow-up outside this map, not a new ticket here.
