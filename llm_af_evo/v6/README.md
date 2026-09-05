# v6 — delta-seed on noisy synthetics: fixing v5's noise-floor problem

## Why v6 exists

v5 built a real multi-domain fitness system (sequential gating,
graded gate-fail tiebreaker, SE-weighted cross-domain combination) and
ran it twice, 150 generations each, on tunable/coatings/mAb. Both runs
converged to a champion. Both champions failed held-out validation —
run1's (`gen41`/`gen148`, a DPP-style quality x diversity ranker) showed
no significant win anywhere and a significant loss on tunable; run2's
(`gen114`, an acquisition + coverage-gap blend) showed no significant win
anywhere and a **significant loss on DTLZ2** (p=0.048), a domain it never
trained on.

Before revising the fitness formula to explain this, we ran a direct
diagnostic on both runs' `history.json`: for every generation where
`best_fitness` improved, is the underlying per-domain score change larger
than the noise floor implied by that domain's own measured campaign-to-
campaign variability at the training campaign count (8 for tunable/
coatings, 16 for mAb)? Domain SEs were estimated from the held-out
validation margin lists (same intrinsic noise, scaled by 1/sqrt(n_train)).

**Result: 1 real-signal event out of 8 "improvements" across 300
generations — and that one (run1 gen41) is a gate-threshold crossing
(fail -> pass), not a domain-quality signal.** Every other apparent
improvement in both runs, including run2's entire post-gate-breakthrough
climb (gen78 -> gen114, which read as "slow, real improvement" in the
moment) is statistically indistinguishable from measurement noise.

This is decisive: no combination-formula fix changes what selection is
climbing if the numbers it's climbing are mostly noise. v6 targets the
noise floor directly, not just the formula on top of it.

## Three bundled changes

Bundling three things that are each independently well-motivated, but
deliberately built and smoke-tested in isolation (same discipline v5
used: pure math layer -> per-domain wiring -> full loop) before
integration, so a failure at any layer is attributable to one change,
not the combination.

### 1. Substrate: noisy synthetics, not noiseless

The obvious first fix for "training signal is noise-dominated" is
noiseless synthetic domains (DTLZ2/ZDT1/ZDT3 already exist as
deterministic oracles). Rejected: this project's own **floor rule**
finding (see `docs/llm_evolved_afs_comprehensive_log.md`) is that on a
noiseless domain, the GP posterior is accurate enough that qLogNEHVI is
already close to optimal — an uncertainty-aware modifier has nothing to
correct. Training on noiseless synthetics would just relocate the
"nothing real to find" problem, not fix it: evolution would converge on
modifier ~= 0, which trivially "passes" every gate and every fitness
check without telling us anything about whether modifiers help in the
noisy regime real domains actually live in.

**Decision: add calibrated Gaussian observation noise to the synthetic
oracles.** This gives both things at once:
- A regime where there's something to find (GP posterior is inaccurate
  under noise, same structural reason real domains are where modifiers
  could plausibly help).
- Cheap, reproducible campaigns, so campaign counts can go to 50+ per
  domain per candidate — driving the *training-time* SE down directly,
  rather than the v5 approach of trying to combine around noise we
  couldn't afford to average out.

Noise is injected as multiplicative Gaussian noise on `query_mo`'s
returned objective values (ground-truth `_Y_raw` stays exact internally,
noise applied only at the observation the AF/GP sees, matching how real
oracles work) — `y_observed = y_true * (1 + N(0, noise_cv))` per
objective. `noise_cv` is a per-oracle-instance constructor parameter, not
a global constant.

Considered and deferred: sweeping noise level (0/15/30/50%) to map the
floor-rule transition as a standalone thesis result. Deferred to *after*
this run, not bundled into it — a sweep is a second question ("how does
signal strength vary with noise") layered on top of the first ("does
delta-seed + calibrated-noise substrate find any real signal at all"),
and answering the first cleanly first keeps this run diagnostic rather
than ambiguous if it fails.

### 2. Delta-seed: evolve a modifier on qLogNEHVI, not a replacement score

v3/v4/v5 all evolved `score_pool(context)` as a full replacement for the
selection score. Every real champion found across all of v2-v5 has, on
inspection, turned out to be "acq_value_norm (real qLogNEHVI) plus a
small additive or blended correction term" — the LLM keeps rediscovering
that EGBO's real acquisition value is most of the signal and only ever
nudges it. Formalizing this as the search space directly (rather than
letting the LLM re-derive it inside an unconstrained rewrite each time)
should narrow what the noise has to distinguish between: "does this
scalar modifier help or hurt" is a much lower-variance question than
"is this whole rewritten scoring function better."

**Contract: `score = baseline_acq_norm + alpha * modifier(context)`.**
`baseline_acq_norm` is `acq_value_norm` (already in context, unchanged).
`alpha` is LLM-set (a literal float in the generated code, same as any
other constant it currently tunes). `modifier(context)` is the only part
the LLM writes.

Additive chosen over multiplicative (`baseline * (1 + modifier)`, harder
to reason about when interpreting what a given modifier value means
against a min-max-normalised baseline) and over gated (`baseline if cond
else modifier`, introduces hand-rolled branching — a documented LLM
failure mode across v2-v5) — additive is the most interpretable and
lowest-risk of the three for a first delta-seed iteration.

### 3. Fitness formula: z-scored mean, no spread penalty

Root cause of v5's `mean - lam*std` pathology (confirmed empirically:
run2's gen78/gen81 lineage had real wins on 2/3 domains and was ranked
*below* gen114, which was uniformly mediocre-negative on all 3, purely
because of lower cross-domain spread):

- **The spread penalty duplicates the gate's job, in the wrong
  direction.** The gate (`ci_lower_16 <= gate_threshold` fails) already
  enforces "not actively harmful anywhere" before a candidate reaches
  the combination step. A symmetric std penalty on top then punishes
  *upside* variance among already-safe candidates — i.e. it punishes
  candidates for having some domains that are unusually good, not for
  having a hidden bad one (the gate's job, already done).
- **Domain-scale imbalance isn't just a measurement-noise problem.**
  v5's `combine_domain_scores_weighted` (inverse-variance weighting)
  fixes the "how uncertain is this domain's estimate" imbalance, but not
  the "how big are real effects on this domain even at the population
  level" imbalance — coatings genuinely swings further than tunable does
  even with infinite campaigns, so raw-margin std conflates "big
  intrinsic domain volatility" with "genuine cross-domain inconsistency."

**Decision: combine per-domain z-scores (`score / se`), mean only, no
std term.** `fitness = mean(z_1, ..., z_k) - gamma * loc`. Z-scoring
addresses the scale-imbalance problem more directly than inverse-
variance-weighting the raw values (a z-score is already unit-free and
comparable across domains regardless of native CV); dropping the std
term removes the redundant-with-the-gate penalty entirely. The gate
itself also switches from a raw-margin threshold (`ci_lower_16 <=
-0.01`, which meant very different things across domains at very
different noise scales) to a z-based threshold (`z <= gate_z_threshold`,
proposed `-1.0`) — consistent with the same "compare in SE units, not
raw-margin units" fix, and directly tied to the diagnostic methodology
that motivated this whole revision.

## Domains

**Train:** ZDT1 (2-obj, convex front), ZDT3 (2-obj, disconnected front),
DTLZ2 with `n_obj=3` (concave front) — three distinct front geometries,
so a genuinely general modifier has to work across shape, not just
overfit one. Noise levels span the low/mid/high range bracketed by
real-domain CVs (coatings ~15%, mAb ~50%): ZDT1 at 15%, DTLZ2-3 at 30%,
ZDT3 at 50%. Gate order is ascending noise (ZDT1 -> DTLZ2-3 -> ZDT3),
keeping the "cheapest/cleanest first" funnel-efficiency reasoning from
v5 (fail fast on the domain that's fastest to get a precise read on).

**Held out, synthetic:** DTLZ2 with `n_obj=5` (higher-dimensional,
never trained on any front of this shape or dimensionality), noise at
35% (deliberately not matching any training level exactly, so this also
tests generalization across noise magnitude, not just structure).

**Held out, real:** mAb + coatings — the actual target domains,
untouched by training or synthetic-noise calibration, exactly as in v5.
Real domains are validation-only, not part of the gating funnel at all
in v6 (a deliberate change from v5, which trained directly on
tunable/coatings/mAb) — the entire point of the noisy-synthetic
substrate is to get a clean training signal, so spending real-domain
campaigns on training rather than reserving all of them for the final
transfer test would be self-defeating.

## Stop criterion

Pre-committed, not adaptive — specifically because both v5 runs saw me
read noise-level fluctuation as "slow, real improvement" in the moment;
a fixed rule removes that judgment call.

**Stop at generation 50.** At that point, validate the champion on the
held-out synthetic domains (DTLZ2-5) with enough campaigns to reach
reasonable power, Wilcoxon-test against baseline:
- No significant win on held-out synthetic at p<0.05: **report a
  negative result** — "delta-seed evolution on calibrated-noise
  synthetics did not find a real generalizing modifier within 50
  generations; training fitness differences were [real per the gen-cap
  diagnostic / still noise-dominated, re-run the same power diagnostic
  used to motivate v6]." Either sub-finding is a legitimate, honest
  thesis result on its own.
- Significant win on held-out synthetic: proceed to validate on mAb +
  coatings. Win transfers to real: real positive result. Win doesn't
  transfer to real: report the nuanced negative result — "modifier
  generalizes within the synthetic noise regime but not to real
  domains" — which is itself informative (says something about what's
  domain-general vs. synthetic-noise-model-specific about the effect).

## Build order (planned, not yet started)

Same layered-verification discipline v5 used:
1. Noisy-oracle layer: add `noise_cv` to `DiscreteSyntheticMOOracle`,
   add a `build_zdt3` classmethod (doesn't exist yet — only
   `build_zdt1`/`build_dtlz2` do), verify noise calibration directly
   (empirical CV of repeated queries at fixed x matches the target).
2. Delta-seed contract: extend `af_interface_v6.py`'s prompt/contract
   docs and one seed program to the new `baseline_acq_norm + alpha *
   modifier(context)` shape; verify a trivial `modifier = 0` seed
   reproduces EGBO's baseline `final_hv` exactly (same kind of
   byte-identical check v4 used for `acq_value_norm`'s introduction).
3. Z-scored fitness module: `multi_domain_fitness_v6.py` (or a v6
   revision of v5's), verified against hand-worked numbers same as v5's
   `combine_domain_scores` was, before wiring to real evaluation.
4. Full loop integration: `evolve_af_v6.py`, smoke-tested end to end on
   a tiny generation count before any real run.

Nothing beyond this document exists yet.
