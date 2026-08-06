## Destination

A locked-down experiment spec for confirming "front-range-normalised sigma UCB (gen6_child0) accelerates early-campaign HV convergence vs. fixed-beta raw-sigma UCB, an advantage that shrinks toward budget-independence as budget grows" — every design decision (primary endpoint, domain-seed count, beta-sweep grid, baseline set, multiple-comparisons handling, real-domain tie-back) resolved and written down precisely enough that an execution session needs to make no further judgment calls.

## Notes

- Domain: LLM-evolved multi-objective BO acquisition functions, MSc thesis extension. Prior work lives in `llm_af_evo/v1_pre_v2/` (tunable_synthetic_oracle.py, sweep_tunable_domain.py, run_tunable_domain_generalization.py, track_front_range.py, track_hv_trajectory.py).
- Existing exploratory evidence (NOT confirmatory — found by scanning batches after the fact, single oracle seed=42, single hand-picked beta=15.0): at budget=20, gen6_child0_tuned beats hint_fixed_ucb head-to-head 15/20 (p=0.008); at budget=40 the lead spikes early (batch 1) then collapses toward statistical noise by the final batch (12/20, p=0.15). This is the pattern the confirmatory spec needs to test rigorously.
- Consult `/grilling` for design-tradeoff tickets, `/research` subagent for literature/statistical-convention or local-codebase-fact tickets.
- Real oracles referenced: `llm_af_evo/shared/ada_coatings_oracle.py` (coatings, 253 real samples), `DiscreteMOExcipientOracle` ("mAb," pool=500, actually synthetic — not real data) — real-domain tie-back ruled out of scope, see map's Out of scope section.

## Decisions so far

- [Primary endpoint metric](issues/01-primary-endpoint-metric.md) — log HV difference vs. true optimum, integrated over batches (AUC), is primary; batches-to-90%-of-optimum is secondary; final HV at budget=40 stays as tertiary (carries the existing null result).
- [Statistical correction method](issues/05-statistical-correction-method.md) — Benjamini-Hochberg (per-beta, across batches) for multiplicity, domain-seed-level cluster bootstrap for CIs, and `statsmodels` MixedLM (domain-seed random effect) instead of repeated Wilcoxon tests once domain-seed replication is added.
- [Domain-seed replication](issues/02-domain-seed-replication.md) — 8 independent domain seeds, domain params fixed across all of them, random-intercept MixedLM primary (random-slope reported as secondary/under-powered at n=8).
- [Beta sweep grid](issues/03-beta-sweep-grid.md) — gen6_child0_tuned swept over beta∈{2,5,10,15,25}, hint_fixed_ucb fixed at beta=2.0, robustness = same-sign in ≥4/5 betas + BH-significant in ≥3/5.
- [Mechanism isolation approach](issues/04-mechanism-isolation-approach.md) — phase_decaying_ucb comparison only (matched-exploration-weight deferred); "indistinguishable" = TOST equivalence test, δ=20% of the primary effect size.

## Not yet specified

- Exact write-up/reporting structure for the confirmatory results (once the spec is locked and results come in) — out of scope for this map, which stops at the spec.

## Out of scope

- Executing the confirmatory experiment itself — this map produces the spec only, not the run.
- Paper/venue positioning, related-work section, write-up structure — destination is the experiment spec, not a publication plan.
- [Real-domain tie-back](issues/06-real-domain-tieback-feasibility.md) — ruled out: coatings (253 real samples) is structurally blocked by mu_sum dominance at any budget; the "mAb" oracle is synthetic (not real data) with a CV=49.7% noise floor on top of an already budget-fragile effect.
