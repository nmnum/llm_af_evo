Type: grilling
Status: open
Blocked by: 01

## Question

How does the confirmatory spec rule out "front-range normalisation's early advantage is just because beta=15 happens to front-load more exploration than beta=2" (the trivial explanation) vs. demonstrating the normalisation mechanism itself matters?

Two candidate designs surfaced:
- **Matched early-exploration-weight comparison**: find a fixed-beta raw-UCB whose batch-1 sigma contribution has the same empirical magnitude as gen6_child0_tuned's sigma_norm*beta at batch 1, then compare full trajectories under that matched condition. If normalised still leads, that's a real mechanism difference.
- **Compare against phase_decaying_ucb** (existing SEED_PROGRAMS entry, explicit progress-decay term): if implicit front-range normalisation and explicit decay produce statistically indistinguishable trajectories, that itself is a publishable positive claim ("implicit self-annealing matches hand-tuned decay without a schedule").

Needs a decision on which (or both) to include in the confirmatory spec, and — for the matched-weight design — exactly how "matched magnitude" is defined/computed (this depends on ticket 01's endpoint choice, since the matching procedure should optimize toward the same metric the confirmatory test uses).
