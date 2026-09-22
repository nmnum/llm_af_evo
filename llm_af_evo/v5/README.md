# v5 — multi-domain fitness

Extends `v4/src/evolve_af_v4.py` (real domain extension of
`evolve_af_v2.py`) with **multi-domain fitness**
(`evaluate_af_multi_domain`, via `src/multi_domain_fitness.py` +
`src/evaluate_multi_domain.py`): selection is no longer against one
domain's fitness alone, since v3/v4 runs that did that (see `v3/README.md`,
`v4/README.md`) generalized poorly to domains they weren't trained on.
See `src/multi_domain_fitness.py`'s module docstring for the
combination/gating design (`mean - LAMBDA_STD_PENALTY * std` of per-domain
`ci_lower_16` margins, sequential cheapest-first gating) and the rationale
for rejecting min-of-domains ("loser's curse") — this design is what v6
later replaced with a z-scored formula after finding v5's runs were mostly
climbing noise (see `v6/README.md`'s "Why v6 exists").

## What's here

- `src/evolve_af_v5.py`, `src/af_interface_v5.py` — the evolution loop
  and seed/hint definitions.
- `src/multi_domain_fitness.py` — pure combination/gating math, kept
  standalone so it could be verified against hand-worked numbers before
  any real GP-campaign evaluation was wired in.
- `src/evaluate_multi_domain.py` — wires real per-domain evaluation
  (`evaluate_af_2b`, domain-generic via `oracle_family`) into the
  gating/combination logic above.
- **Train:** tunable, coatings, mAb (`ORACLE_DEFAULTS` in
  `evolve_af_v5.py`) — the three real/controlled domains, gated
  cheapest-noise-first.
- **Held out, never trained on:** DTLZ2 and ZDT1 — synthetic, fully
  deterministic domains, used to check whether a formula generalizes
  past the three training domains' shared structure.
  `experiments/generate_dtlz2_training_set.py` /
  `generate_zdt1_training_set.py` build their fixed campaign sets;
  `experiments/training_logs_{dtlz2,zdt1}/{train,heldout}/` are the
  generated logs.
- `experiments/evolution_runs/run1/`, `run2/` — two full 150-generation
  real-LLM runs. Both reached `checkpoint.json` generation 150;
  `best_fitness_ever` was slightly positive for run1 (~0.0036) and
  slightly negative for run2 (~-0.0029). Neither run's held-out
  generalization has been re-analyzed here since v6 diagnosed both as
  mostly noise-driven (see `v6/README.md`) — treat these numbers as
  pointers, not a conclusion.
