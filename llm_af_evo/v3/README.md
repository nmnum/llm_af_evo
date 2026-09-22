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
  import it without reaching into `v1_pre_v2`. Added
  `from_fixed_realization()` (see Status item 2 below); otherwise
  unmodified.
- `llm_af_evo/shared/full_replay.py` — added an `oracle_family="tunable"`
  branch to `reconstruct_oracle` (see `_TUNABLE_DEFAULTS`).
- `experiments/generate_tunable_training_set.py` — training-set generator
  (see Status item 1).
- `experiments/run_2b_diagnostic_v3.py` — `oracle_family`-generalized copy
  of `v1_pre_v2/experiments/run_2b_diagnostic.py` plus a new Step C
  seed-noise check (see Status item 3).
- `experiments/training_logs_tunable/{train,heldout}/` — 8 generated
  campaigns, checked in as a starting set.
- `data/concrete/concrete_dataset.csv` — UCI Concrete Slump dataset (Yeh
  2007), pulled from the EGBO paper's own repo
  ([andrelowky/CMOO-Algorithm-Development](https://github.com/andrelowky/CMOO-Algorithm-Development),
  MIT licensed), `Real-world datasets/concrete_dataset.data`. 103 samples,
  7 input variables (cement, slag, fly ash, water, superplasticizer,
  coarse/fine aggregate), 3 output properties (Slump, Flow, 28-day
  Compressive Strength).

## Status: DONE (this section) / validated with one caveat

1. **Training-set generator built**: `experiments/generate_tunable_training_set.py`.
   Defaults (`plateau_sharpness=5.0`, `noise_level=0.08`,
   `noise_mode="proportional"`, `scale2=3.0`) are the exact configuration
   already measured favorable in `tunable_domain_generalization_results.json`.
   `experiments/training_logs_tunable/{train,heldout}/*.json` — 8 campaigns
   (6 train / 2 heldout), real config sizes (`budget=40`, `n_init=10`,
   `batch_size=5`, `pool_size=256`) — is checked in as a starting set;
   regenerate/extend with `--n_seeds` for a real run.

2. **`reconstruct_oracle` noise-determinism gap — FIXED.**
   `TunableSyntheticMOOracle.from_fixed_realization()` (new classmethod)
   accepts an already-fixed `Y_raw` directly instead of `__init__`'s normal
   behavior of drawing a fresh noisy realization from `(Y_true, seed)`.
   Verified directly: two independent `reconstruct_oracle()` calls on the
   same log now produce byte-identical `Y_raw`/`Y_true`. Requires the log
   to carry `oracle_Y_true` (noiseless ground truth) alongside
   `oracle_Y_raw` — `generate_tunable_training_set.py` dumps both.

3. **`run_2b_diagnostic_v3.py` built and run against the real pipeline** —
   `oracle_family`-threaded copy of `v1_pre_v2/experiments/run_2b_diagnostic.py`,
   plus a new Step C (seed-noise check, mirroring `mab_noise_diagnostic.py`'s
   Axis 3: same fixed campaign, N repeat seeds, paired margin vs. baseline).

   **Timing**: ~0.5-0.6s/batch, ~3.6s/campaign — much cheaper than excipient
   (~6.6-9.3s/campaign), since `d=6`/`pool_size=256` here vs. excipient's
   `d=16`. Projected cost for a real evolution run (8 campaigns × 40
   children) is **~20 minutes**.

   **Seed-noise floor — validated, with one important nuance.** First pass
   using `trust_only` as the reference AF gave a paired-margin std of
   6.61% across 8 seeds, driven almost entirely by ONE outlier seed
   (-19.6% margin; without it, std=1.89%). Investigated directly (pulled
   both runs' per-batch decision logs, no exceptions/fallback-to-random on
   either side) — **not a pipeline bug**: `trust_only` is pure exploitation
   with no uncertainty term, and on that one seed it greedily locked onto a
   poorly-GP-estimated high-mean region from batch 0 onward and never
   recovered (hv_trajectory essentially flat: 8.65→8.77 over 6 batches),
   while the baseline's novelty-aware exploration found a much better
   front on the identical log/seed (10.56→10.91). Re-ran Step C with a
   simple UCB-style AF (mean + 2×normalised-uncertainty, the same
   worked-example formula in `evolve_af_v3.py`'s system prompt) instead:
   **std=0.64% across all 8 seeds, no outliers.** That's the real noise
   floor for this domain — ~40x cleaner than mAb's measured 25.57%
   seed-margin-std, and for the first time in this project, SMALLER than
   the ~1% margin gaps evolution is trying to detect (every other domain
   tried so far has this backwards by 20-45x).

   **Takeaway for future evolution runs on this domain**: `trust_only`
   itself is a fragile, high-variance seed AF here (not because of noise —
   because pure mu-only exploitation is a genuinely weak strategy that can
   get stuck) — expect it to occasionally produce a real outlier campaign,
   and don't mistake that for domain noise when reading results. Any
   analysis of this domain's own noise floor should use a
   uncertainty-aware AF as the reference, not `trust_only`.

   **Cross-campaign CV — now pinned down, after fixing a real design bug.**
   The n=2 estimate (11.2%) was too small a sample to trust, so a 64-seed
   set (48 train / 16 heldout) was generated to get a real number — which
   came back at **23.3%**, much worse than expected. Root cause: the
   generator built a FRESH `TunableSyntheticMOOracle` per campaign
   (reasoning: "genuinely different pools, not resampled subsets"), but
   this domain's Pareto-optimal region requires ALL of `x2..xd` (5 of 6
   dims) simultaneously near zero — a sparse target for a random 256-point
   pool — so different random draws land meaningfully different distances
   from it purely by chance (`final_hv` ranged 7.7-19.2 across the 16
   heldout campaigns). That's real pool-QUALITY variance, not domain
   noise, and the validated 1.5-2.9% number was never measured under that
   design (`run_tunable_domain_generalization.py`, the script that
   produced it, builds ONE oracle once and only varies initial-point
   draws — same pattern as coatings' one-fixed-real-pool design).
   `generate_tunable_training_set.py` now matches that: one shared oracle
   (`--tunable_seed`, default 0) built once, `make_shared_inits` varies
   only the initial points. Regenerated the same 64-seed set under the
   fix and re-measured: **CV=1.2%** (`final_hv` now 10.65-11.02 across all
   16 heldout campaigns) — matches the original validated range.

4. **Concrete Slump dataset — NOT validated, likely fails the same audit
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
5. The AgNP self-driving-lab dataset (same EGBO-paper repo,
   `AgNP self-driving lab/Results_Algo*_Run15.csv`) has too few real
   campaigns to build an independent training/held-out split the way
   excipient/coatings do; the repo also ships a GP-fitted 256-point
   virtual emulator (`virtual_x_gp_256.csv`/`virtual_con_256.csv`) that
   could be used as a Method-2-style ("empirical surrogate benchmark")
   oracle if pursued, with that method's usual surrogate-artifact caveat.

6. **First real evolution run (`experiments/evolution_runs/run1`, 30
   generations, `pop_size=12`, `n_offspring=2`, real LLM) — champion found,
   plus a mode-collapse bug found and fixed.**

   Champion `gen5_child0` (13 LOC, an adaptive-UCB idea: progress-weighted
   blend of front-range-normalized GP mean and std) reached mean_margin
   +0.47%, win_rate 0.88, and was never beaten again — **25 of the
   remaining 25 generations were stagnant**.

   Reading all 72 logged `af_code_logs/call_*.py` docstrings in order
   showed why: generations 0-15 covered genuinely diverse ideas, but from
   ~generation 8 onward the LLM anchored on ONE family — "resample the GP
   posterior under noise and count dominance/hypervolume-expansion
   frequency" — and kept re-issuing near-verbatim restatements of it for
   the rest of the run (several docstrings identical across 10+
   generations apart). That family's bootstrap `ci_lower_16` was
   consistently 2-5x lower than the champion's, so the fitness metric
   correctly rejected it every time — the bug wasn't in the fitness
   metric, it was that the LLM never tried anything else. Root cause:
   `llm_propose_child`'s stagnation-annealing note (see its own docstring)
   suggests a fixed list of example "structurally different" mechanisms
   once stagnant, and the model latched onto one suggested exemplar
   without rotating through the others as intended.

   **Fixed**: added `MECHANISM_FAMILIES` + `classify_mechanism_families()`
   and a per-stagnation-streak `family_attempt_counts` (persisted through
   checkpoint/resume, reset on any fitness improvement) to
   `evolve_af_v3.py`. The annealing note now explicitly forbids whichever
   family has already been retried `FAMILY_REPEAT_LIMIT` (2) times this
   streak and leads with the least-tried remaining ones, forcing rotation
   instead of repetition. Verified via a mock-mode end-to-end smoke test
   (checkpoint round-trip) and direct prompt inspection (both the
   "one family worn out" and "all families worn out" note variants, plus
   the classifier correctly tagging run1's actual dominant-family
   docstrings while leaving the champion's UCB idea unclassified).

## Recommended next step

Both noise sources are validated: seed-noise std=0.64% (UCB-style
reference AF), cross-campaign CV=1.2% (n=16, shared-oracle design). Run1
found a real, robust champion but also exposed and got a fix for a
stagnation mode-collapse bug (item 6) — worth a second real evolution run
to see whether the fix lets later generations actually diversify past the
gen5 champion instead of plateauing again. Treat concrete/AgNP as
secondary, gated on their own audits (item 4/5).

## Status: run2 exists, not yet written up here

`experiments/evolution_runs/run2/` (68 generations, real LLM,
post-mode-collapse-fix) has been run — `checkpoint.json` shows
`best_fitness_ever` around -0.0005, i.e. essentially flat/negative
relative to run1's champion (+0.0003). Champion program
(`run2/best_af.py`) is a progress-weighted exploit/explore blend, same
general family as run1's. This result hasn't been analyzed or written up
in this document yet — treat the numbers above as a pointer for whoever
picks this up next, not a conclusion.
