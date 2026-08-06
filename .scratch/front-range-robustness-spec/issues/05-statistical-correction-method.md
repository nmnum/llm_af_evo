Type: research
Status: resolved

## Question

What multiple-comparisons correction and confidence-interval method should the confirmatory spec pre-commit to, given the design will test across several batches/domain-seeds/beta-values?

Specifically need a recommendation (with justification, not just a name) on:
- Bonferroni vs. Benjamini-Hochberg (or another FDR-control method) for the batch x domain-seed x beta grid of tests.
- Bootstrap CI construction for the primary endpoint (resampling campaigns within each domain-seed vs. resampling domain-seeds themselves vs. a hierarchical bootstrap) — given campaigns are nested within domain-seed pools (ticket 02), a naive flat bootstrap over all campaigns pooled across domains would understate variance.
- Whether a paired or hierarchical/mixed-effects test (e.g. mixed model with domain-seed as random effect) is more appropriate than repeated Wilcoxon signed-rank tests once domain-seed replication (ticket 02) is added — this determines whether scipy.stats.wilcoxon remains sufficient or a different tool (e.g. statsmodels mixedlm) is needed.

Resolve via a research subagent citing standard references for BO-benchmark statistical practice (e.g. how comparable acquisition-function benchmarking papers report significance) plus general multiple-comparisons/bootstrap literature — not just picking a convention unjustified.

## Answer

Use Benjamini-Hochberg (FDR control) per-beta across batches (not Bonferroni over the whole grid), since the batch/domain-seed/beta family is a trend-survey (per Demšar 2006 and BoTorch/Ax benchmarking convention), not a set of independent high-stakes claims needing FWER control. Use a cluster bootstrap resampling domain-seeds (keeping all campaigns per resampled domain-seed) rather than a flat pooled-campaign bootstrap, since campaigns are nested within domain-seed pools. Once domain-seed replication (ticket 02) exists, replace repeated per-domain-seed Wilcoxon tests with a `statsmodels` `MixedLM` (domain-seed as random intercept, ideally random slope on condition) to avoid pseudoreplication. Full justification and citations: [research file](05-statistical-correction-method-research.md).
