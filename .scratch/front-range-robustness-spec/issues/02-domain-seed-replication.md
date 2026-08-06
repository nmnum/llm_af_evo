Type: grilling
Status: open

## Question

How many independent oracle-pool seeds (TunableSyntheticMOOracle.build(seed=...)) should the confirmatory run use, and how should they be selected?

All existing evidence uses a single pool instantiation (seed=42) with 20 campaign-init seeds drawn from it — the effect might be a property of that one pool draw, not of front-range normalisation generally. Needs a decision on:
- Number of independent domain seeds (e.g. 5? 10?) — tradeoff against compute budget (~1s/batch, n_campaigns x n_batches x n_domains x n_conditions).
- Whether domain params (plateau_sharpness=5.0, noise_level=0.08, scale2=3.0) stay fixed across domain seeds, or whether a small grid around the sweep's chosen working point should also vary (risk of scope creep into re-running sweep_tunable_domain.py's job).
- How results aggregate across domain seeds for the final test: pooled Wilcoxon, or a hierarchical/mixed-effects model treating domain-seed as a random effect (the more statistically correct choice given campaigns are nested within domain seeds).
