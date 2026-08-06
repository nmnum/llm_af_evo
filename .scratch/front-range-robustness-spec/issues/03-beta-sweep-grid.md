Type: grilling
Status: resolved

## Question

What beta grid should the confirmatory sweep use for gen6_child0, and what counts as "the effect is robust across beta" vs. "it's a knife-edge at the one calibrated value"?

beta=15.0 was hand-picked to match the domain sweep's measured dominance_ratio (~11-17); it has never been swept. Needs a decision on:
- The grid itself (e.g. 1, 5, 10, 15, 20, 30 — or log-spaced around dominance_ratio's measured range).
- Whether hint_fixed_ucb's beta=2.0 also needs to be swept for a fair comparison (right now one side is tuned-to-domain, the other is an arbitrary literature-standard default — this asymmetry itself may need addressing, see ticket 04).
- The robustness criterion: e.g. "primary endpoint's confirmatory result holds (same sign, still significant after correction) for at least N of the M beta values tested" — needs a concrete N/M, not just "looks robust."

## Answer

**Beta grid for `gen6_child0_tuned`: {2, 5, 10, 15, 25}** (5 values) — log-ish spacing spanning below/at/above the measured `dominance_ratio≈11-17` range. 2 anchors at literature-standard UCB's own beta (tests whether normalisation helps even without domain-specific tuning), 15 is the already-tested calibrated value, 25 tests over-shooting. Cost: `8 domains × 20 campaigns × ~6.5 batches × (1 hint_fixed_ucb + 5 beta variants) ≈ 85-105 min` total, affordable as a background run.

**`hint_fixed_ucb`'s beta stays fixed at 2.0** (its literature-standard default) — not swept in this ticket. Sweeping both sides here would balloon to a full 5×M grid and conflates two different questions; "is gen6_child0 robust to its own beta" (this ticket) is answered fairly against the baseline as normally deployed. Any hint-side tuning (e.g. to find a matched-exploration-weight comparison) is deferred to ticket 04 (mechanism isolation), which needs its own methodology for that anyway.

**Robustness criterion (two-part)**: (a) same-sign effect favoring `gen6_child0_tuned` on the primary endpoint (AUC of log HV difference) in **at least 4 of the 5 beta values**, and (b) BH-corrected significance in **at least 3 of the 5**. Splitting direction from significance avoids over-punishing beta values far from the calibrated regime (e.g. beta=2 may legitimately underperform without invalidating the claim) while still requiring more than a one-off fluke.
