Type: grilling
Status: open

## Question

What beta grid should the confirmatory sweep use for gen6_child0, and what counts as "the effect is robust across beta" vs. "it's a knife-edge at the one calibrated value"?

beta=15.0 was hand-picked to match the domain sweep's measured dominance_ratio (~11-17); it has never been swept. Needs a decision on:
- The grid itself (e.g. 1, 5, 10, 15, 20, 30 — or log-spaced around dominance_ratio's measured range).
- Whether hint_fixed_ucb's beta=2.0 also needs to be swept for a fair comparison (right now one side is tuned-to-domain, the other is an arbitrary literature-standard default — this asymmetry itself may need addressing, see ticket 04).
- The robustness criterion: e.g. "primary endpoint's confirmatory result holds (same sign, still significant after correction) for at least N of the M beta values tested" — needs a concrete N/M, not just "looks robust."
