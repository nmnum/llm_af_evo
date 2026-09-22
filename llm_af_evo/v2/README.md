# v2 — minimal-seed / hint-driven AF design

Variant of `v1_pre_v2/src/evolve_af.py` using `af_interface_v2.py`'s
minimal-seed design: one hand-written seed (`trust_only`, kept only to
anchor contract syntax/conventions) plus one-line `STRATEGY_HINTS` that
real-LLM mode's generation-0 bootstrap implements itself, instead of
shipping the 7 hand-written seed implementations `v1_pre_v2/src/evolve_af.py`
used. See `src/evolve_af_v2.py`'s own docstring for the full reasoning
(references `research_seed_population.md`'s FunBO/FunSearch/AlphaEvolve
literature read) and for why fitness is full-campaign replay
(`full_replay.run_2b_campaign`) rather than the single-step proxy the
original `evolve_af.py` used.

## What's here

- `src/evolve_af_v2.py`, `src/af_interface_v2.py` — the minimal-seed
  evolution loop and its seed/hint definitions.
- `experiments/evolution_runs/run_v2_mAb_dro_hvi/` — one run on the mAb
  domain. `history.json` has `"mock": true` — this is a **mock-mode**
  run (fast smoke test against a stub instead of the real LLM), not a
  real evolution result. Champion (`best_af.py`) is a trivial pure-
  exploitation sum-of-means scorer.
- `experiments/fixed_merge_hints/` — three manually-patched AF programs
  (`call_000{16,21,30}_fixed.py`).
- `experiments/pilot_*_mab.py` + matching `*_results.json` — three MAB
  (multi-armed-bandit) pilot scripts (decay-schedule, fixed-hints,
  merge-hints variants) and their results.
- `experiments/diagnose_mu_sigma_dominance.py`,
  `experiments/diagnose_win_overlap.py`,
  `experiments/mab_noise_diagnostic.py`,
  `experiments/validate_population_2b.py`,
  `experiments/validate_precise.py` — standalone diagnostic scripts.

No real-LLM evolution run exists for v2 in this repo — treat the mock run
above as a pipeline smoke test, not a result.
