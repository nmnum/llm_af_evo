Type: grilling
Status: open

## Question

What is the single pre-specified primary endpoint for the confirmatory experiment?

Candidates surfaced so far:
- **batches-to-X%-of-asymptotic-HV** (e.g. batches to reach 90% of that condition's own final HV) — directly tests "reaches the territory earlier."
- **AUC / integrated regret** of the HV trajectory over the full budget — standard BO-literature metric, captures the whole shape (early lead + late convergence) in one number instead of picking a threshold.
- Some combination: AUC as primary, batches-to-90% as a secondary/interpretability metric.

Needs a decision on: which one is primary (drives the confirmatory test's power calculation and the pre-registration), what X% threshold if using batches-to-X%, and whether final-HV-at-long-budget is kept as a secondary endpoint (it already has a negative/null result worth reporting alongside).
