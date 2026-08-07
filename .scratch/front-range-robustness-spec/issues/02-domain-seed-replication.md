Type: grilling
Status: resolved

## Question

How many independent oracle-pool seeds (TunableSyntheticMOOracle.build(seed=...)) should the confirmatory run use, and how should they be selected?

All existing evidence uses a single pool instantiation (seed=42) with 20 campaign-init seeds drawn from it — the effect might be a property of that one pool draw, not of front-range normalisation generally. Needs a decision on:
- Number of independent domain seeds (e.g. 5? 10?) — tradeoff against compute budget (~1s/batch, n_campaigns x n_batches x n_domains x n_conditions).
- Whether domain params (plateau_sharpness=5.0, noise_level=0.08, scale2=3.0) stay fixed across domain seeds, or whether a small grid around the sweep's chosen working point should also vary (risk of scope creep into re-running sweep_tunable_domain.py's job).
- How results aggregate across domain seeds for the final test: pooled Wilcoxon, or a hierarchical/mixed-effects model treating domain-seed as a random effect (the more statistically correct choice given campaigns are nested within domain seeds).

## Answer

**8 independent domain seeds** (`TunableSyntheticMOOracle.build(seed=...)`, 8 distinct values) — enough levels for MixedLM (ticket 05) to estimate a domain-seed random-intercept variance component reliably (rule-of-thumb minimum ~5, comfortably more at 8-10), affordable at ~15-20 min total compute (8 domains × 20 campaigns × ~6-7 batches × 2 conditions at ~0.85-1.15s/batch).

**Domain params stay fixed across all 8 seeds** (plateau_sharpness=5.0, noise_level=0.08, scale2=3.0 — the sweep's chosen working point) — only the pool-generating `seed` varies. Keeps this ticket cleanly isolated to "replicates under a fixed, already-validated domain regime," not conflated with a param sweep (that's ticket 03's job for beta, and `sweep_tunable_domain.py` already covers domain-param selection).

**Aggregation model**: `AUC ~ condition + (1 | domain_seed)` (random-intercept MixedLM) as the **primary** confirmatory test. `AUC ~ condition + (condition | domain_seed)` (random-slope) reported as a **secondary/exploratory** check only — explicitly flagged as under-powered at n=8 domain-seeds (random-slope variance estimation conventionally wants ~20+ groups per Gelman & Hill / glmm-FAQ conventions; 8 is sufficient for intercept-only but not slope). Considered bumping to 20 seeds specifically to promote random-slope to primary; decided against — not worth the added scope for what the confirmatory spec needs (the core question is "does the AUC advantage replicate," which random-intercept already answers cleanly).
