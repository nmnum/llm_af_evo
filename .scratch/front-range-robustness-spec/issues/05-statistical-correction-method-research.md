Type: research
Status: resolved (research artifact for ticket 05)

# Multiple-comparisons correction, bootstrap CI, and hierarchical-test recommendation

## Recommendation summary

1. **Use Benjamini-Hochberg (FDR control), not Bonferroni**, for the batch x domain-seed x beta grid of primary-endpoint tests — but restructure the grid first so the correction is applied to the smallest defensible family of *genuinely separate* hypotheses, not to every batch/seed/beta cell independently.
2. **Use a cluster (not flat, and not fully hierarchical two-stage) bootstrap that resamples domain-seeds as the primary resampling unit**, retaining all campaigns within each resampled domain-seed pool, for the primary-endpoint CI. This respects the nesting (campaigns nested in domain-seed pools) without introducing unnecessary extra variance inflation from also resampling within-domain campaigns.
3. **Once domain-seed replication (ticket 02) is added, replace repeated per-domain-seed Wilcoxon signed-rank tests with a linear mixed-effects model** (`statsmodels.MixedLM`, or `mixedlm` formula API) with domain-seed as a random effect (random intercept, and ideally a random slope on acquisition-function condition), condition/beta/batch as fixed effects. This is the correct way to avoid pseudoreplication once there is more than one domain-seed: treating each domain-seed's 20 campaigns as 20 independent Wilcoxon-test observations, repeated once per domain-seed, either (a) throws away the domain-seed-level uncertainty entirely (if you just run Wilcoxon once on campaigns pooled across all domain-seeds) or (b) requires ad hoc meta-analytic combination across domain-seed-level Wilcoxon test statistics (if you run per-domain-seed Wilcoxon tests and combine p-values) — both are worse-justified than a single mixed model that has the correct error structure built in.

---

## 1. Bonferroni vs Benjamini-Hochberg

### The core issue is what "the grid" actually means statistically

The ticket frames this as "batch x domain-seed x beta" cells needing correction. Before choosing FDR vs FWER, this family needs to be examined for what kind of multiplicity it actually represents, because Bonferroni and BH answer different questions:

- **Bonferroni / FWER control** answers "what is the probability that *any* of my declared 'significant' findings is a false positive?" It is the right choice when a single false positive anywhere would undermine the paper's central claim — i.e., when the tests are not independent replications of the same claim but a set of *distinct* claims, each of which the reader might act on individually (e.g., "beta=5 differs from beta=20 AND batch 3 differs from batch 7 AND domain-seed 1 differs from domain-seed 4" — a grab-bag of unrelated assertions).
- **Benjamini-Hochberg / FDR control** answers "of the tests I declare significant, what proportion are expected to be false positives?" It is appropriate when the tests form a coherent family being surveyed together for a pattern (which batches/betas show the effect), and the paper's claim is a *summary* over the family ("the effect holds early and shrinks by late batches") rather than a conjunction of every individual cell being independently load-bearing.

This experiment's confirmatory claim (per map.md) is exactly the latter: "AF accelerates early-campaign HV convergence, an advantage that shrinks toward budget-independence as budget grows." That is a *trend* claim across batches, replicated across domain-seeds and beta settings — not N independent point claims. The batch dimension in particular is not really "N independent comparisons" — it is N looks at one trajectory, which is the textbook case for FDR control (Benjamini & Hochberg 1995, JASA — the original FDR paper explicitly frames FDR as suited to screening a family of related hypotheses where some true effects are expected, at the cost of tolerating a controlled fraction of false positives among the declared significant results). Bonferroni's FWER guarantee is tuned for the opposite situation — very few or zero true effects expected, and any single false positive is costly (safety-critical/clinical framing) — which does not describe a benchmarking sweep over batches/betas whose purpose is precisely to map out *where* an effect appears and disappears.

Additionally, Bonferroni becomes severely underpowered as the grid grows (several batches x several domain-seeds x several betas easily reaches 50-200+ cells), and BO acquisition-function benchmarking is already effect-size-starved (typical head-to-head win-rate margins are modest, as evidenced by the ticket's own exploratory numbers: 15/20, p=0.008 at budget=20 collapsing to 12/20, p=0.15 by the final batch). Applying Bonferroni across the full grid would very likely wash out real, correctly-timed effects (e.g. the "early advantage" signal) purely from lost power, defeating the experiment's purpose of characterizing *when* the effect appears/disappears rather than just whether it exists anywhere.

### Precedent from ML/BO benchmarking practice

The most directly relevant methodological precedent is Demšar (2006), "Statistical Comparisons of Classifiers over Multiple Data Sets" (JMLR 7, https://jmlr.org/papers/v7/demsar06a.html) — the standard reference for how ML benchmark papers handle "compare method A vs B across many replicate settings" (there: datasets; here: domain-seeds x betas x batches). Demšar recommends the Wilcoxon signed-rank test for pairwise two-classifier comparison across replicate settings, and explicitly notes that "some form of correction for multiple testing is required to hold the overall alpha level" when multiple such comparisons/settings are surveyed together — and in practice the ML-benchmarking literature that follows this convention overwhelmingly uses FDR-style or rank-based (Friedman + Nemenyi) corrections rather than Bonferroni, precisely because Bonferroni's power loss across dozens of dataset/setting combinations is considered unacceptable. BoTorch/Ax-based acquisition-function benchmarking practice (e.g. typical BoTorch tutorials and papers such as the multi-objective acquisition-function comparisons referenced by the honegumi BoTorch benchmarking docs, https://honegumi.readthedocs.io/en/latest/curriculum/tutorials/benchmarking/benchmarking.html) generally reports effect sizes with confidence bands over many seeds (recommended 25+ seeds) rather than running one significance test per grid cell at all — significance testing across many seeds/settings in this literature is typically a secondary confirmatory step layered on top of visual/CI-based comparison, further supporting an FDR (survey-oriented) rather than FWER (single-decision) framing.

### Concrete recommendation

- Apply **Benjamini-Hochberg** within each natural sub-family, not across the entire flattened batch x domain-seed x beta grid indiscriminately. Concretely:
  - For a fixed beta, treat the set of per-batch comparisons (across domain-seeds, if testing at each batch using the mixed-model approach in §3) as one BH family — this directly controls the FDR over the "which batches show a significant effect" question, which is exactly the shrinking-advantage claim being tested.
  - Treat different beta values as *separate* families (i.e., run BH per-beta across batches) rather than pooling beta into the same correction family, since beta represents qualitatively different acquisition-function configurations being screened, and pre-registering per-beta family boundaries avoids post-hoc family-size gaming.
  - Domain-seed should not be a unit needing its own correction if the mixed-effects model (§3) is used — domain-seed becomes a random effect absorbed into one model per batch-beta cell, not a separate test to correct for.
- Pre-register the exact family boundaries (which axis is corrected jointly vs. treated as separate families) before running the confirmatory experiment, since the choice of family materially changes power and must not be chosen post hoc after seeing results.
- Report both raw and BH-adjusted p-values (q-values) plus effect sizes/CIs — p-values alone, corrected or not, should not be the sole basis for the "shrinks toward budget-independence" claim; the trend should be visible in the CI/effect-size trajectory independent of significance thresholding.

---

## 2. Bootstrap CI construction under nesting

### The nesting problem

Campaigns (e.g. 20 campaign-init seeds) are nested within domain-seed pools (independent synthetic oracle instantiations). A **flat bootstrap** that pools all campaigns across all domain-seeds and resamples individual campaigns with replacement treats each campaign as an independent unit of variation. But campaigns within the same domain-seed share the same underlying oracle/Pareto-front instantiation, so they are correlated (not exchangeable with campaigns from a different domain-seed pool). Flat bootstrapping under this dependence structure understates the true between-domain-seed variance component and produces CIs that are too narrow — the same failure mode as pseudoreplication/pooling clustered data in a naive t-test.

The general remedy, per the clustered/hierarchical bootstrap literature (e.g. "Estimating Uncertainty in Classifier Performance with Applications to ... Nested Data," https://arxiv.org/pdf/2606.26422; ClusterBootstrap R package documentation, https://link.springer.com/article/10.3758/s13428-019-01252-y; lmeresampler for nested LMM bootstrap, https://journal.r-project.org/articles/RJ-2023-015/) is:

- **Cluster bootstrap**: resample domain-seeds (clusters) with replacement; for each resampled domain-seed, retain *all* of its original campaigns (no within-cluster resampling). This reflects only the domain-seed-level sampling variation.
- **Hierarchical (two-stage) bootstrap**: resample domain-seeds with replacement, *and* independently resample campaigns with replacement within each resampled domain-seed. This reflects both domain-seed-level and campaign-level sampling variation and is more conservative (wider CIs) than the cluster bootstrap.

### Recommendation: cluster bootstrap (domain-seed as the resampling unit), not flat, and hierarchical only if within-domain campaign-count variability is itself a concern

Given the design (several domain-seeds, ~20 campaigns each, campaigns are re-initializations of the *same* fixed acquisition-function/oracle-pool comparison rather than an independently-varying measurement process), the recommended procedure is:

1. Resample domain-seeds with replacement (B domain-seeds drawn from the actual domain-seed set, with repeats allowed).
2. For each resampled domain-seed, keep its full original set of campaigns unchanged (do not additionally resample within-domain-seed).
3. Compute the primary endpoint (batches-to-X%-of-asymptotic-HV, or AUC of HV trajectory) aggregated over this bootstrap sample.
4. Repeat many times (≥2000 resamples recommended given a modest domain-seed count, since percentile/BCa CIs need a reasonably fine resample distribution) to build the bootstrap distribution, then take percentile or BCa CIs.

This is the standard **cluster bootstrap** (not full hierarchical), and it is the more defensible default here because:
- The primary source of *independent* extra-sample uncertainty being asked about ("would this hold up on a different domain instantiation?") is the domain-seed draw, not additional resampling noise in which 20 of the 20 already-run campaigns happen to get drawn — campaign count per domain-seed is fixed by design, not a random sample size whose sampling variability needs separate estimation.
- Additionally bootstrapping within-domain-seed campaigns (full hierarchical bootstrap) mainly adds value when the number of within-cluster units is itself small/variable and its sampling noise is a first-order concern; with ~20 campaigns per domain-seed already averaged, the within-domain-seed component contributes comparatively little extra uncertainty relative to the between-domain-seed component, so the extra conservatism of hierarchical bootstrap is unlikely to change conclusions materially and mainly costs power/precision.
- If in practice this experiment has few domain-seeds (single digits), a hierarchical bootstrap should be used as a **sensitivity check** — with few clusters, cluster-bootstrap CIs can be badly behaved (the literature on cluster-robust variance/cluster bootstrap consistently flags that a small number of clusters, e.g. <10, invalidates the classical asymptotic justification for percentile CIs) — recommend explicitly reporting the domain-seed count and, if it is small, using a wild cluster bootstrap or explicitly flagging CI width as approximate rather than nominal-coverage.

Do **not** use a flat bootstrap over pooled campaigns as the primary CI method — it directly reproduces the pseudoreplication problem described in the pseudoreplication references below and will materially understate uncertainty, risking overstated confidence in the "shrinks toward budget-independence" claim.

---

## 3. Mixed-effects model vs. repeated Wilcoxon signed-rank tests

### Why repeated per-domain-seed Wilcoxon tests become inadequate once domain-seed replication is added

`scipy.stats.wilcoxon` on a single domain-seed's 20 paired campaigns is a reasonable within-domain-seed paired test. But once ticket 02 adds several independent domain-seeds, there are two bad options if Wilcoxon is kept as the sole tool:

- **(a) Pool all campaigns across domain-seeds and run one Wilcoxon test**: this is a pseudoreplication error — it treats campaigns from different domain-seeds as exchangeable independent replicates of the same paired comparison, inflating the effective N and understating the true variance (a domain-seed-level random effect is silently ignored), producing artificially small p-values. This is the same failure mode flagged in the general pseudoreplication/mixed-model literature (e.g. "Addressing Pseudoreplication in Linear Mixed Models," https://vsni.co.uk/case-studies/dealing-with-pseudo-replication-in-linear-mixed-models; Arnqvist, "Mixed Models Offer No Freedom from Degrees of Freedom," https://arnqvist.org/tree_2020.pdf — noting the correction only works if the random-effects structure is specified correctly, i.e. simply switching to a mixed model without a matching random effect for domain-seed does not itself fix pseudoreplication).
- **(b) Run one Wilcoxon test per domain-seed and then need a second-stage combination** (meta-analytic p-value combination, or majority-vote across domain-seeds) to get a single overall conclusion — this is workable but statistically clunky: it discards the magnitude/precision information in each domain-seed's estimate (Wilcoxon gives only a p-value/rank statistic, not a directly poolable effect size with SE), and any combination rule (Stouffer's method, Fisher's method, vote-counting) is a weaker, more ad hoc substitute for a model that has the right variance structure built in from the start.

### Recommended model form

A linear mixed-effects model (`statsmodels.regression.mixed_linear_model.MixedLM`, or the `mixedlm` formula API `statsmodels.formula.api.mixedlm`) with domain-seed as a random effect is the correct tool, per the standard mixed-model-for-nested-data argument (e.g. the pseudoreplication references above, and standard mixed-model texts such as the "Learning Statistical Models Through Simulation in R" chapter on LMMs with one random factor, https://psyteachr.github.io/stat-models-v1/linear-mixed-effects-models-with-one-random-factor.html — "using random effects that identify each stratum ... provide[s] the correct degrees of freedom and therefore correct statistical inference").

Suggested minimal model, per batch (or with batch as an additional fixed effect / interaction if pooling across batches within one model):

```
endpoint_ijk ~ condition_i + beta_j + condition_i:beta_j + (1 + condition_i | domain_seed_k)
```

in `mixedlm` / R-lme4-style notation:

- **Fixed effects**: `condition` (AF vs. baseline — the acquisition function comparison), `beta` (if pooling multiple beta values into one model rather than fitting per-beta as in the FDR-family recommendation above), and their interaction (`condition:beta`) if beta-dependence of the effect is itself of interest.
- **Random effects**: a random intercept for `domain_seed` at minimum (`(1 | domain_seed)`), and ideally a random slope for `condition` within `domain_seed` (`(1 + condition | domain_seed)`) if there is reason to think the AF-vs-baseline advantage itself varies systematically by domain-seed pool (which the design's whole premise — testing robustness across independently-instantiated oracle pools — suggests is plausible and worth allowing for, rather than assuming a fixed effect that's constant across domains).
- Campaigns within a domain-seed remain the base observational unit (residual/observation-level variance), correctly nested below the domain-seed random effect — this gives each campaign appropriate within-domain-seed weight while domain-seed-level variation is modeled explicitly rather than ignored or manually corrected for post hoc.

This should be fit **once domain-seed replication exists (i.e., >1 domain-seed)** — with a single domain-seed (current exploratory status), Wilcoxon on paired campaigns remains adequate and a mixed model with one random-effect level is not meaningfully different from (and cannot even usefully estimate variance for) a fixed-effect-only model. The switch to mixed-effects should happen at the same point ticket 02's domain-seed replication is introduced, not before.

### Relationship to the FDR-correction recommendation in §1

If the mixed model is fit once per (batch, beta) cell (as recommended in §1, to keep beta as separate correction families and batch as the within-family axis), the `condition` fixed-effect p-value from each model becomes the per-batch test statistic that gets BH-corrected within its beta family — this cleanly combines both recommendations: the model handles domain-seed nesting/pseudoreplication correctly at the per-cell level, and BH handles the multiplicity across batches within each beta's family.

---

## Sources cited

- Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." JASA. (Original FDR framing — appropriate for surveying families of related hypotheses expecting some true effects, vs. FWER control for isolated high-stakes decisions.)
- Demšar, J. (2006). "Statistical Comparisons of Classifiers over Multiple Data Sets." JMLR 7. https://jmlr.org/papers/v7/demsar06a.html — standard ML-benchmarking reference recommending Wilcoxon signed-rank for pairwise comparison across replicate settings and flagging the need for multiple-testing correction across settings; establishes the convention this ticket's design most closely resembles.
- BoTorch/Ax acquisition-function benchmarking conventions (honegumi BoTorch benchmarking tutorial, https://honegumi.readthedocs.io/en/latest/curriculum/tutorials/benchmarking/benchmarking.html) — illustrates the field norm of reporting effect/CI trajectories over many seeds (25+ recommended) rather than per-cell significance testing as the primary comparison method, supporting an FDR/survey framing over per-cell FWER control.
- Cluster/hierarchical bootstrap for nested data: "Estimating Uncertainty in Classifier Performance with Applications to Large Language Models and Nested Data" (arXiv:2606.26422), https://arxiv.org/pdf/2606.26422; ClusterBootstrap R package / Springer paper, https://link.springer.com/article/10.3758/s13428-019-01252-y; lmeresampler (nested LMM bootstrap), https://journal.r-project.org/articles/RJ-2023-015/.
- Pseudoreplication and mixed-effects correction: "Addressing Pseudoreplication in Linear Mixed Models," https://vsni.co.uk/case-studies/dealing-with-pseudo-replication-in-linear-mixed-models; Arnqvist, "Mixed Models Offer No Freedom from Degrees of Freedom," https://arnqvist.org/tree_2020.pdf (caveat: mixed models only fix pseudoreplication if random-effects structure matches the true clustering).
- `statsmodels` `MixedLM` / `mixedlm` formula API — standard Python tool for fitting linear mixed-effects models with random intercepts/slopes, referenced in ticket as the candidate tool once domain-seed replication is added.
