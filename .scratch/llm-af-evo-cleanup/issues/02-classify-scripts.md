Type: grilling
Status: open
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
