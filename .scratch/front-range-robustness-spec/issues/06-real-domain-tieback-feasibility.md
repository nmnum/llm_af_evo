Type: research
Status: resolved

## Question

Is a real-domain (coatings and/or mAb) tie-back of the "early-convergence advantage" finding feasible, and at what budget?

The confirmatory spec's value is much higher if it can show the same batches-to-X%/AUC pattern (even non-significantly) on a real oracle at a realistic short budget, closing the loop with the thesis's original coatings (mu_sum-dominated) / mAb (noise-floor-dominated CV~49.7%) diagnosis. Needs local-codebase facts, not a design decision:

- Pool sizes of `DiscreteADACoatingsOracle` (`llm_af_evo/shared/ada_coatings_oracle.py`) and the mAb oracle — how many real samples exist, and what's the largest n_campaigns x budget combination that doesn't exhaust/overlap the pool across campaigns (recall `make_shared_inits`/`_queried` semantics from `excipient_campaign_mo.py`).
- Given coatings' mu_sum-dominance and mAb's noise floor (both diagnosed as suppressing the sigma-term's influence in the original thesis work), what budget range would even give the early-convergence pattern a chance to show up at all on either real domain — i.e. is this tie-back actually testable, or does the same failure mode that blocked the original confirmation also block a short-budget replication?

Resolve via a research subagent reading the two oracle files plus the original thesis diagnosis in `docs/llm_evolved_afs_comprehensive_log.md`, reporting concrete pool sizes/budget bounds and a recommendation on whether ticket inclusion of a real-domain tie-back is worth the confirmatory spec's scope, or whether it should be ruled out of scope for this map.

## Answer

Ruled out of scope. Coatings (253 real samples, `DiscreteADACoatingsOracle`) is blocked by mu_sum dominance, a budget-independent structural property (7/8 AFs rank-equivalent, sigma barely moves rankings) — no budget window fixes this. The "mAb" oracle (`DiscreteMOExcipientOracle`, pool=500) is not real data — a tunable synthetic simulator, not fit to real measurements — and carries a CV=49.7% noise floor on top of an already budget-fragile effect. Full detail and recommendation in `06-real-domain-tieback-feasibility-research.md`.
