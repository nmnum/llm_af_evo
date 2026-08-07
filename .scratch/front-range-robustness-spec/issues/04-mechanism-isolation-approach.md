Type: grilling
Status: resolved
Blocked by: 01

## Question

How does the confirmatory spec rule out "front-range normalisation's early advantage is just because beta=15 happens to front-load more exploration than beta=2" (the trivial explanation) vs. demonstrating the normalisation mechanism itself matters?

Two candidate designs surfaced:
- **Matched early-exploration-weight comparison**: find a fixed-beta raw-UCB whose batch-1 sigma contribution has the same empirical magnitude as gen6_child0_tuned's sigma_norm*beta at batch 1, then compare full trajectories under that matched condition. If normalised still leads, that's a real mechanism difference.
- **Compare against phase_decaying_ucb** (existing SEED_PROGRAMS entry, explicit progress-decay term): if implicit front-range normalisation and explicit decay produce statistically indistinguishable trajectories, that itself is a publishable positive claim ("implicit self-annealing matches hand-tuned decay without a schedule").

Needs a decision on which (or both) to include in the confirmatory spec, and — for the matched-weight design — exactly how "matched magnitude" is defined/computed (this depends on ticket 01's endpoint choice, since the matching procedure should optimize toward the same metric the confirmatory test uses).

## Answer

**Only `phase_decaying_ucb` comparison included; matched-exploration-weight deferred/out of scope for this spec.** `phase_decaying_ucb` already exists in `SEED_PROGRAMS`, needs no calibration pass, and rides along the existing 8-domain × 20-campaign × 5-beta grid for ~15-20 min marginal cost. Matched-exploration-weight would require its own per-domain-seed calibration sub-experiment (8 more design decisions nested inside this one) with real risk of reviewer pushback on the matching procedure's own arbitrariness — flagged as a follow-up worth a separate effort if the phase_decaying_ucb result is inconclusive, not part of this confirmatory spec.

**"Statistically indistinguishable" is defined via TOST (two one-sided tests) equivalence testing**, not ordinary non-significance — plain non-significant p-values are absence-of-evidence, not evidence-of-equivalence, which would undercut the whole point of this ticket. Equivalence margin **δ = 20% of the primary `gen6_child0_tuned` vs. `hint_fixed_ucb` AUC effect size** (self-referential to the experiment's own measured effect, rather than an external/arbitrary anchor) — independently confirmed as landing on standard convention from two traditions: FDA/EMA bioequivalence's "80-125% rule" (~20-25% margin) and Lakens' TOST methodology (Cohen's d=0.2 as smallest-effect-size-of-interest). Sources: http://daniellakens.blogspot.com/2016/12/tost-equivalence-testing-r-package.html, https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/Equivalence_Tests_for_the_Difference_Between_Two_Proportions.pdf
