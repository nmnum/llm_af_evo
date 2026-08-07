Type: grilling
Status: resolved

## Question

What is the single pre-specified primary endpoint for the confirmatory experiment?

Candidates surfaced so far:
- **batches-to-X%-of-asymptotic-HV** (e.g. batches to reach 90% of that condition's own final HV) — directly tests "reaches the territory earlier."
- **AUC / integrated regret** of the HV trajectory over the full budget — standard BO-literature metric, captures the whole shape (early lead + late convergence) in one number instead of picking a threshold.
- Some combination: AUC as primary, batches-to-90% as a secondary/interpretability metric.

Needs a decision on: which one is primary (drives the confirmatory test's power calculation and the pre-registration), what X% threshold if using batches-to-X%, and whether final-HV-at-long-budget is kept as a secondary endpoint (it already has a negative/null result worth reporting alongside).

## Answer

**Primary endpoint: log hypervolume difference vs. the true Pareto front, integrated over batches (AUC)** — `log(HV_true - HV_observed(batch))`, averaged/integrated across the campaign's batches. This is literature-standard for MOBO benchmarking (BoTorch's own multi-objective tutorial plots exactly this quantity per iteration with CIs across seeds — see https://botorch.org/docs/tutorials/multi_objective_bo/ and https://botorch.org/docs/multi_objective), avoids the arbitrary-threshold noise-sensitivity problem of a pure batches-to-X% metric, and is computable here because the synthetic domain has a known analytic Pareto front (HV_true is fixed per domain-seed).

Normalization is against the **true optimum**, not each campaign's own final HV — the self-normalized alternative was considered specifically for portability to a real-domain tie-back, but ticket 06 (real-domain tie-back feasibility) ruled that out of scope, removing the reason to prefer it. True-optimum normalization also matches literature convention directly.

**Secondary endpoint: batches-to-90%-of-true-optimum-HV** — kept for interpretability in the write-up ("reaches 90% of optimal N batches sooner"), threshold set at 90% (informal enough to avoid over-sensitivity to late-campaign noise near full convergence given this domain's achieved_cv~0.06, but early enough to be meaningfully different from the asymptote).

**Tertiary endpoint: final HV at long budget (budget=40)** — kept, not dropped, because it already carries a negative/null result (12/20 wins, p=0.15) that grounds the "advantage shrinks toward budget-independence" half of the destination's claim; reported explicitly as tertiary so it can't be mistaken for a primary confirmatory claim on its own, addressing ticket 05's multiple-comparisons concern by keeping the endpoint hierarchy explicit rather than treating all three as equally-weighted tests.
