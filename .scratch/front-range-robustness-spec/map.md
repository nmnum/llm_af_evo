## Destination

A locked-down experiment spec for confirming "front-range-normalised sigma UCB (gen6_child0) accelerates early-campaign HV convergence vs. fixed-beta raw-sigma UCB, an advantage that shrinks toward budget-independence as budget grows" — every design decision (primary endpoint, domain-seed count, beta-sweep grid, baseline set, multiple-comparisons handling, real-domain tie-back) resolved and written down precisely enough that an execution session needs to make no further judgment calls.

## Notes

- Domain: LLM-evolved multi-objective BO acquisition functions, MSc thesis extension. Prior work lives in `llm_af_evo/v1_pre_v2/` (tunable_synthetic_oracle.py, sweep_tunable_domain.py, run_tunable_domain_generalization.py, track_front_range.py, track_hv_trajectory.py).
- Existing exploratory evidence (NOT confirmatory — found by scanning batches after the fact, single oracle seed=42, single hand-picked beta=15.0): at budget=20, gen6_child0_tuned beats hint_fixed_ucb head-to-head 15/20 (p=0.008); at budget=40 the lead spikes early (batch 1) then collapses toward statistical noise by the final batch (12/20, p=0.15). This is the pattern the confirmatory spec needs to test rigorously.
- Consult `/grilling` for design-tradeoff tickets, `/research` subagent for literature/statistical-convention or local-codebase-fact tickets.
- Real oracles referenced: `llm_af_evo/shared/ada_coatings_oracle.py` (coatings), mAb oracle (see repo root / shared dir) — sizes/budgets not yet confirmed, see ticket on real-domain tie-back feasibility.

## Decisions so far

- [Statistical correction method](issues/05-statistical-correction-method.md) — Benjamini-Hochberg (per-beta, across batches) for multiplicity, domain-seed-level cluster bootstrap for CIs, and `statsmodels` MixedLM (domain-seed random effect) instead of repeated Wilcoxon tests once domain-seed replication is added.

## Not yet specified

- Exact write-up/reporting structure for the confirmatory results (once the spec is locked and results come in) — out of scope for this map, which stops at the spec.

## Out of scope

- Executing the confirmatory experiment itself — this map produces the spec only, not the run.
- Paper/venue positioning, related-work section, write-up structure — destination is the experiment spec, not a publication plan.
