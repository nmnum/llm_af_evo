Type: grilling
Status: resolved

## Question

What target directory structure should `llm_af_evo/` have? e.g. buckets like
`src/` (current pipeline modules), `experiments/` (pilot/diagnostic/validate
runners), `archive/` (superseded/dead-end code and docs), `data/` (results +
run logs), `docs/` — or some other convention that fits how the thesis will
cite this work. This decision fixes the target buckets that ticket 02's
script classification sorts into.

## Answer

Era-first split, mirroring the two-part history SESSION_LOG.md and
llm_evolved_afs_comprehensive_log.md already narrate, with pipeline vs.
one-off scripts separated *within* each era:

```
llm_af_evo/
  v1_pre_v2/
    src/           # current-for-that-era pipeline: evolve_af.py, af_interface.py, ...
    experiments/   # pilots/diagnostics from that era (compose_*, da_coreg*, etc.)
  v2/
    src/           # evolve_af_v2.py, af_interface_v2.py, evolve_af_2b.py, ...
    experiments/   # run_*_pilot.py, diagnose_*, validate_* from Jul 28-31
  shared/          # fitness_common.py, oracle files used by both eras
  data/            # evolution_runs/, logs/, training_logs*/, result JSONs — NOT split by era
  docs/            # SESSION_LOG.md, llm_evolved_afs_comprehensive_log.md, L_COATINGS_FINDINGS.md, research_*.md — NOT split by era
```

Rationale: era is the primary navigation axis (matches how the thesis will
likely narrate this work chronologically), but within an era, pipeline code
and one-off pilot/diagnostic scripts are kept apart since pilots are
citation evidence rather than reusable code. `data/` and `docs/` stay as
single shared top-level buckets rather than being split by era, because
existing docs already cross-reference across the v1/v2 boundary (e.g.
llm_evolved_afs_comprehensive_log.md is explicitly a companion document to
SESSION_LOG.md) and splitting them would fragment those cross-references.

No `archive/` bucket — "superseded" files (e.g. `af_interface.py` vs
`af_interface_v2.py`) are not pulled into a separate archive; they live in
their era's `src/` or `experiments/`, since the era split already makes
supersession legible without an extra bucket. Ticket 02 (script
classification) sorts each of the ~50 scripts into one of: `v1_pre_v2/src`,
`v1_pre_v2/experiments`, `v2/src`, `v2/experiments`, or `shared/`.
