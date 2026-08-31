# v3 — synthetic-domain screening + real-world generalization

Started from `af-evolution` branch conversation on adding a third,
lower-noise evaluation domain alongside excipient (noise-dominated,
CV~50%) and coatings (mu_sum-dominated) — see `evolve_af_v2.py`'s and
`full_replay.py`'s own docstrings for the diagnosis of those two domains'
failure modes.

## What's here

- `src/evolve_af_v3.py` — copy of `v2/src/evolve_af_v2.py`, extended only
  to add `oracle_family="tunable"` to `ORACLE_DEFAULTS`/
  `N_FITNESS_SEEDS_DEFAULTS`. Everything else (fitness metric, bootstrap
  CI, adaptive gamma, LLM prompt scaffold) is unchanged from v2.
- `src/af_interface_v3.py` — unmodified copy of `v2/src/af_interface_v2.py`
  (SEED_PROGRAMS/STRATEGY_HINTS).
- `llm_af_evo/shared/tunable_synthetic_oracle.py` — moved here from
  `v1_pre_v2/src/` so `full_replay.py` (which lives in `shared/`) can
  import it without reaching into `v1_pre_v2`. Unmodified otherwise.
- `llm_af_evo/shared/full_replay.py` — added an `oracle_family="tunable"`
  branch to `reconstruct_oracle` (see `_TUNABLE_DEFAULTS` and the branch's
  own comment for the **known gap**: `TunableSyntheticMOOracle.__init__`
  draws a fresh noise realization from `(Y_true, seed)` rather than
  accepting an already-fixed `Y_raw` the way the real oracles do, so
  replay determinism across separate `reconstruct_oracle` calls for this
  family is not yet guaranteed — needs either a v3-specific alternate
  constructor or accepting non-determinism until fixed).
- `data/concrete/concrete_dataset.csv` — UCI Concrete Slump dataset (Yeh
  2007), pulled from the EGBO paper's own repo
  ([andrelowky/CMOO-Algorithm-Development](https://github.com/andrelowky/CMOO-Algorithm-Development),
  MIT licensed), `Real-world datasets/concrete_dataset.data`. 103 samples,
  7 input variables (cement, slag, fly ash, water, superplasticizer,
  coarse/fine aggregate), 3 output properties (Slump, Flow, 28-day
  Compressive Strength).

## Still open / not yet built

1. **A training-set generator for the tunable domain** (analogous to
   `generate_training_set.py`/`generate_coatings_training_set.py`) —
   `evolve_af_v3.py --oracle tunable` will fail today at
   `load_training_campaigns` for lack of training-log JSON files in the
   expected schema (`X_init`/`Y_init`/`oracle_X_raw`/`oracle_Y_raw`/
   `budget`/`n_init`/`batch_size`, plus this domain's own
   `tunable_scale1`/`tunable_scale2`/`tunable_noise_level`/
   `tunable_noise_mode`/`tunable_boundary_gain`/`tunable_seed` config keys —
   see `full_replay.py`'s `_TUNABLE_DEFAULTS` comment).
2. **The `reconstruct_oracle` noise-determinism gap** noted above.
3. **Concrete Slump dataset — NOT validated, likely fails the same audit
   coatings failed.** Checked directly against the project's own
   precedent (`ada_coatings_oracle.py`'s rationale for dropping a
   redundant objective): `slump` and `flow` are themselves nearly
   redundant (r=+0.906, same magnitude as coatings' rejected
   `xrf_conductance`/`conductivity` pair at r=1.0). Worse, the Pareto
   front stays tiny under every objective pair and every min/max
   direction convention tried — best case (`flow=max, strength=min`) is
   **8.7% (9/103)**, well short of coatings' post-fix 31.2%. This is the
   same near-degenerate-MO-problem shape that got the original 3-objective
   coatings oracle rejected. **Do not build a v3 evolution run on this
   dataset without first running the equivalent of
   `diagnose_mu_sigma_dominance.py` on it** — the raw front-size check
   above is a necessary-but-not-sufficient red flag, not a full audit.
   Kept in the repo as a documented, parked option, not a recommended
   next domain.
4. The AgNP self-driving-lab dataset (same EGBO-paper repo,
   `AgNP self-driving lab/Results_Algo*_Run15.csv`) has too few real
   campaigns to build an independent training/held-out split the way
   excipient/coatings do; the repo also ships a GP-fitted 256-point
   virtual emulator (`virtual_x_gp_256.csv`/`virtual_con_256.csv`) that
   could be used as a Method-2-style ("empirical surrogate benchmark")
   oracle if pursued, with that method's usual surrogate-artifact caveat.

## Recommended next step

Build (1) — the tunable-domain training-set generator — before anything
else here, since it's the one domain with an actual measured, favorable
noise/mu-dominance profile
(`tunable_domain_generalization_results.json`: CV~1.5-2.9% vs. excipient's
~50%). Treat concrete/AgNP as secondary, gated on their own audits.
