# Approach L: LLM-Evolved Acquisition Functions — Findings and Generalization Test

## Research question

Can an LLM evolve acquisition functions (AFs) — short, closed-form, per-candidate
scoring functions — that beat a sophisticated, jointly-optimized baseline
(`mo_egbo_novelty`: qLogNEHVI + U-NSGA-III evolutionary candidate diversity +
novelty-weighted batch selection) on short-horizon, noisy, multi-objective
physical-domain optimization?

## Part 1 — mAb formulation domain (16D categorical, synthetic oracle)

### The methodological arc

The investigation went through several rounds of *finding a way the pipeline
could be wrong, and either fixing it or ruling it out* before any result was
trusted. In order:

1. **Naive single-step counterfactual regret (2a) as training fitness.**
   Score a candidate AF by whether its per-step pick would have beaten the
   baseline's actual pick on that step, using logged campaign snapshots.
   Validated as non-degenerate via a true-oracle-HV diagnostic (exploration
   genuinely earns more than pure exploitation, confirmed with real oracle
   ground truth) and a λ-sweep for a GP-only proxy version (λ*=10000).

2. **Fitness gaming, found and fixed.** Reusing the GP-only exploration-credit
   proxy as the *training* signal (not just an evaluation diagnostic) let a
   degenerate AF — pure uncertainty, ignoring predicted quality entirely —
   score near-optimally by construction, because the proxy is dominated by
   the exploration term at the λ that made it non-degenerate as a diagnostic.
   Fixed by switching the training signal to true-oracle HV comparison
   (legitimate offline, since training data is synthetic; the *deployed* AF
   itself never sees the oracle).

3. **Rank-equivalence collapse, found and fixed.** A full 20-generation,
   160-real-LLM-call run produced *zero* fitness movement — traced to every
   "different" evolved program being a positive-scalar reparametrization of
   the same underlying ranking (campaign-state terms that are constant across
   the whole pool at a given step don't change `argsort` order, so they're
   rank-inert). Fixed with (a) a selection-signature dedup that rejects
   children provably rank-equivalent to an existing population member, and
   (b) an explicit prompt warning about this exact trap.

4. **2a proxy confirmed leaky.** Spearman(2a fitness, true full-campaign HV)
   = 0.125, p = 0.645 across the run1 population — statistically
   indistinguishable from *no* correlation. The 2a-selected "best" AF scored
   *below* the baseline under full-campaign replay. Single-step regret is not
   a valid proxy for campaign-level performance on this short a horizon.

5. **Switched to full-campaign replay (2b) as training fitness**, matching
   FunBO's approach: run each candidate AF as the actual acquisition function
   for a complete multi-batch campaign, replay against training campaigns
   (`training_logs/train`, disjoint from the held-out validation set),
   compare final hypervolume against the baseline on the same campaigns.

6. **Interface fixed to be self-documenting and generalization-safe.**
   `score_pool(context)` takes a nested dict (`gp_posterior` keyed by
   objective name, `pareto_front`, `ref_point`, `campaign` state) rather than
   positional arrays — eliminates the "remember which column is which, and
   that viscosity is pre-flipped" failure mode. The prompt was extended with
   (a) a description of what the baseline actually does, (b) what
   `pareto_front`/`ref_point` are *for* (hypervolume-improvement reasoning,
   described conceptually — not naming EHVI/qNEHVI as branded methods, to
   avoid inviting rediscovery-by-label rather than genuine derivation), (c)
   what the fitness measures and the train/eval gap, (d) the actual (not
   assumed) batch-selection mechanism.

### Result on mAb

A 20-generation, 160-real-LLM-call run under the corrected (2b) fitness,
with the corrected prompt, produced 40 distinct candidate AFs
(0 rank-equivalent duplicates in every generation). Verified via code-log
audit: 34/38 unique programs used `pareto_front`, 22/38 used `ref_point`,
36/38 used the GP mean (exploitation) — the model had and used the intended
vocabulary, in most attempts combining multiple signals, not just restating
one.

**None beat the baseline on 20-campaign held-out validation.** Best result
(a `trust_only`-family exploitation AF): **+2.4%, p = 0.50** — not
significant. `ehvi_approx` (a hand-written, hypervolume-improvement-flavored
seed using `pareto_front`/`ref_point`): **+1.0%, p = 0.70**.

**Interpretation:** this is a negative result with a specific, ruled-out set
of alternative explanations — not "the LLM is dumb" or "the search didn't
try." The search space was shown to be adequate (`ehvi_approx` and the
general shape of what evolution found both use the intended structure); the
fitness was shown to be valid (2b, not the leaky 2a); the search was shown
to be diverse (rank-dedup + prompt fix eliminated the collapse mode); 40
distinct, vocabulary-informed attempts still landed at a tie, not a win.

## Part 2 — Generalization test: ADA coatings (MacLeod et al. 2022)

### Why coatings, and what had to be checked before trusting it

MacLeod et al. 2022 ("Self-driving laboratories can advance the Pareto front
for material properties," *Nat. Commun.*) is a real self-driving-lab dataset,
structurally different from the mAb domain in every dimension that matters
for testing whether the design rule generalizes: 4D continuous inputs (not
16D categorical), real experimental noise (not synthetic), a different
physical system (spray-combustion Pd films, not protein excipients).

Data: 253 real samples pooled across 4 real campaigns
(`github.com/berlinguette/ada`), used as a **discrete real-data pool with
nearest-neighbour snap-query — no GP fitting, no extrapolation**. This was a
deliberate design choice, not a convenience: an earlier phase of this same
research area explicitly documents that a naive GP surrogate fit to this
lab's coatings data produced values 193× the empirical maximum
(`synthetic_coatings_oracle.py`'s docstring, and confirmed independently via
the `sdl-adaptive` repo's own README: *"GP oracle extrapolated to 193× the
empirical maximum in 7D with 91 points"*). Snap-to-real-data sidesteps that
failure mode by construction.

**Two problems were found and fixed before any result was trusted:**

1. **The objectives weren't genuinely multi-objective.** The dataset's three
   natural output columns (conductance, XRF-normalized conductance,
   conductivity) are not three independent trade-offs. Checked directly:
   `xrf_conductance` and `conductivity` correlate at **r = 1.0000** —
   essentially the same measurement up to linear rescaling. The resulting
   3-objective "Pareto front" was 5/253 points (2.0%) — a near-degenerate
   single-objective problem in a multi-objective disguise. **Corrected** to a
   genuine 2-objective problem: conductivity (maximize) vs. conductance
   position-to-position standard deviation (minimize, a uniformity proxy).
   Checked: r = 0.0032 (genuinely independent), Pareto front 79/253 (31.2%)
   — a real trade-off structure. The original 3-objective framing's results
   were discarded entirely, not folded into any conclusion.

2. **Objective-direction coupling was hardcoded to the mAb oracle's
   `["max","max","min"]` throughout the shared campaign-running
   infrastructure** (`run_mo_campaign`, `strategy_mo_egbo_novelty`,
   `strategy_evolved_af`, `pareto_front_of`, `to_allmax`) — would have
   silently corrupted every HV/Pareto computation on an all-max oracle like
   coatings. Fixed by making every one of these read `oracle.objective_directions()`
   instead, with the old module-level constant preserved as a default so
   every existing mAb call site is bit-for-bit unaffected (verified:
   `DiscreteMOExcipientOracle.objective_directions()` returns the exact same
   list that was previously hardcoded).

3. **Run-to-run non-determinism, found and fixed.** Two nominally identical
   20-campaign runs (same seed, same code) produced p = 0.033 ("significant
   win") and p = 0.82 ("null") respectively — traced through three layers:
   `torch.manual_seed` alone is insufficient for CPU reproducibility (BLAS
   thread-count env vars must be set before numpy/scipy/torch are imported
   anywhere in the process, not after); and pymoo's `UNSGA3` was never
   passed an explicit `seed=`, leaving its internal RNG uncontrolled. All
   three fixed. Confirmed via same-seed sanity check that the campaign
   *generation* pipeline is now reproducible; genuine across-seed variance
   (which real Pareto front a campaign discovers, out of many reachable
   ones on a sparse discrete pool) remains and is real, not a bug — handled
   by reporting the pattern across independent replicates rather than
   trusting any single run's p-value.

### AFs tested on coatings

- `trust_only` — pure exploitation (parameterized to loop over
  `context["objective_names"]` instead of hardcoded `Tm`/`kD`/`viscosity`;
  regression-tested to reproduce the original hardcoded version's output
  exactly on mAb data, max diff 7×10⁻¹⁵, before trusting it on a different
  oracle).
- `ehvi_approx` — hypervolume-improvement-flavored, uses `pareto_front`/
  `ref_point` (parameterized and regression-tested the same way, exact
  match).
- `phase_decaying_ucb` — hand-designed adaptive: explore-weighted early,
  exploit-weighted late, novelty boosted under stagnation (parameterized).
- `evolved_adaptive` — the actual LLM-evolved winner from the mAb 2a run
  (not a hand-written guess): uncertainty weighted *more* as campaign
  progress increases, exploitation weighted less (parameterized).

### Result on coatings (3 independent replicates, 20 campaigns × 5
conditions each, distinct seed blocks per replicate)

| AF | Mean % diff vs. baseline | Pattern across 3 replicates |
|---|---|---|
| `trust_only` | +4.8% | positive in 3/3, significant (p<0.05) in 2/3 |
| `phase_decaying_ucb` | +4.6% | strongly significant positive in 2/3, null (not negative) in 1/3 |
| `evolved_adaptive` | +4.4% | positive in 3/3, significant in 1/3 |
| `ehvi_approx` | +0.0% | genuinely mixed sign, never significant either direction |

## Combined finding

- **Front-based, hypervolume-improvement-style AFs (`ehvi_approx`) tie the
  baseline consistently across both domains** — mAb (+1.0%, p=0.70) and
  coatings (+0.0% mean across replicates, never significant). This is the
  most domain-general result: this specific AF family reliably matches a
  sophisticated MC-integrated, jointly-optimized acquisition strategy,
  never beats it, on two structurally different real-world problems.

- **Exploitation-flavored AFs are domain-sensitive.** Tie on mAb (+2.4%,
  p=0.50); show a modest, replicated edge on coatings (~+4.5%, significant
  in 2/3 independent replicates, binomial p=0.0072 for "significant in ≥2 of
  3 replicates by chance alone").

- **Adaptivity provides no measurable benefit over static exploitation, in
  either domain.** On coatings specifically — where there was room to show
  an effect, since exploitation itself has a real edge there — both the
  hand-designed adaptive AF and the real LLM-evolved adaptive AF land within
  noise of plain `trust_only` (+4.6%, +4.4%, +4.8% respectively, mean
  differences smaller than each individual run's own variance). Knowing
  *when* to shift between exploring and exploiting added nothing detectable
  beyond just committing to exploitation from the first batch, on these
  40-experiment-budget campaigns.

## What is explicitly *not* claimed

An earlier draft of this write-up over-claimed a causal mechanism
("landscape sparsity and dimensionality" driving the domain difference) and
cited two specific numbers in support of it. Both were checked directly
against the saved replicate data and did not hold up (a claimed "~18
reachable Pareto fronts, top 5 = 58%" measured as 12–79 distinct levels and
30–87% depending on rounding granularity, never landing near the claimed
figures at any reasonable threshold; a claimed "ehvi_approx 15.4% CV, 2× the
baseline" measured as 8.8% vs. a baseline of 8.2% — no 2× gap in either
direction). **The mechanism behind the mAb/coatings difference is explicitly
unresolved** — plausible candidates include dimensionality, pool sparsity,
real vs. synthetic noise, and continuous vs. categorical inputs, all of
which differ simultaneously between the two domains tested, so none can be
isolated from n=2. This is flagged as a specific, well-scoped open question
for future work, not asserted as a settled finding.

## Methodological contributions (transferable beyond this project)

1. **Single-step counterfactual regret is not a valid training signal for
   acquisition-function evolution on short-horizon campaigns** — it can be
   actively misleading, not merely noisy (Spearman ρ=0.125 against the true
   objective, and a documented case where the proxy-selected "best" AF
   scored below baseline under the real objective). Full-campaign replay,
   though far more expensive, is necessary.
2. **A GP-only exploration-credit fitness, validated as non-degenerate as a
   diagnostic, is not safe to reuse as a training/selection signal** — the
   same properties that make it useful for detecting exploration value make
   it gameable by degenerate exploration-only candidates when used to rank
   a population.
3. **Rank-equivalence is a distinct failure mode from lack of diversity** —
   an evolutionary search can produce many source-code-distinct individuals
   that are behaviorally identical (same `argsort` order under the fixed
   selection rule), silently consuming the entire search budget. Detectable
   cheaply via a selection-signature hash, orthogonal to diffing source code.
4. **CPU-based BoTorch/pymoo pipelines are not reproducible under
   `torch.manual_seed` alone** — BLAS thread-count environment variables
   (set before any numpy/scipy/torch import in the process) and explicit
   `seed=` arguments to library-internal optimizers (e.g. pymoo's `UNSGA3`,
   which silently defaults to an uncontrolled RNG) are both necessary. This
   was found via a same-seed sanity check that should be standard practice
   before trusting any p-value from a similar pipeline.
5. **Discrete-pool, snap-to-real-data oracles avoid the extrapolation
   failure mode of fitted surrogates on small real datasets** — validated
   against a previously-documented 193× extrapolation artifact on a related
   dataset from the same research area.

## Part 3 — The composition gate: does batch *selection* explain the tie?

L's whole architecture holds one thing fixed regardless of which AF is
evolved: batch selection is always pure top-k over per-candidate scores
(`select_batch`). This leaves a specific, testable alternative to "rewrite
the optimiser" as the next step: maybe the tie isn't about per-candidate
scoring quality at all, but about what happens *after* scoring — a jointly
diverse batch might realize value that top-k on the same scores throws
away. This is also the one lever genuinely untested by prior art: LLaMEA-BO
(van Stein & Bäck 2025) already demonstrates "LLM rewrites the full BO
algorithm," but only in a sequential q=1 setting with no batch composition
step to speak of — so it doesn't test this specific hypothesis either.

**Design**: `strategy_evolved_af_novelty` — identical candidate generation
to `strategy_evolved_af`, but the final selection step is swapped from
top-k to the baseline's own novelty-weighted greedy selection
(`novelty_aware_select_vectorised`, real defaults `w_acq=0.9`/`w_nov=0.1`,
the same function `mo_egbo_novelty` itself uses). Run as a 5-condition,
3-independent-replicate comparison (baseline, `trust_only`×{top-k,
novelty}, `ehvi_approx`×{top-k, novelty}) on both domains, using the same
BLAS-env-var and pymoo-`UNSGA3`-seed determinism fixes validated in Part 2.

**mAb result**: `trust_only` novelty-vs-topk is flat — −0.0%/−0.1%/−0.3%
(std 0.1) across all 3 replicates, never significant. `ehvi_approx`
novelty-vs-topk is noisy and directionally inconsistent (−3.5%/+0.7%/+5.8%,
std 3.8), never significant. But mAb is the domain where L already ties
baseline regardless of AF (`trust_only` +2.4% p=0.50, `ehvi_approx` +1.0%
p=0.70) — a null selection effect here is ambiguous: it's consistent with
"selection method doesn't matter" but equally consistent with "there was no
gap for selection to modulate in the first place."

**Coatings result (the discriminating test — a real, replicated gap exists
here on top-k)**: `trust_only_novelty` shows **+5.1% mean** (+4.9%/+2.6%/
+8.0%, all 3 replicates directionally positive, 2/3 individually
significant) — essentially unchanged from `trust_only_topk`'s **+4.8%**
(+7.5%/+1.1%/+5.8%, same 2/3-significant pattern). The direct
novelty-vs-topk comparison confirms this: −2.5%/+1.5%/+2.0% (mean +0.4%,
std 2.0), inconsistent direction, never significantly positive across 3
replicates. `ehvi_approx` shows the same null pattern from the other side
(novelty-vs-topk: −1.6%/+0.1%/−1.8%, mean −1.1%, inconsistent, never
significant).

**Gate verdict: closed, negative.** Swapping L's fixed top-k selection for
the baseline's own joint diversity-weighted selection does not close, open,
or otherwise move the gap between evolved-AF scoring and baseline, on
either domain, and the coatings result — where there was an actual effect
to modulate — settles the ambiguity the mAb-only result left open. The
tie is not an artifact of batch selection.

**Why the null is this clean, mechanistically**: `trust_only` scores are
peaked — summed GP posterior means concentrate merit tightly around the
predicted optimum, so the top-5-by-score and the novelty-weighted-5 are
essentially the same batch; there's little room for a diversity bonus to
reorder candidates that scoring has already separated by a wide margin.
The clustering failure mode novelty selection exists to prevent doesn't
manifest when scoring has already committed to a region. `ehvi_approx`'s
noisier-but-still-null novelty-vs-topk numbers fit the same picture from
the other side: front-based HV-improvement scores are flatter across the
candidate pool, so novelty selection *does* reorder the batch more — the
reorder just doesn't produce a replicated directional effect either way.

## Part 4 — The generation gate: does candidate *proposal* explain the tie?

The one lever Part 3 flagged as untested: candidate generation itself.
`strategy_evolved_af` always builds its pool from qLogNEHVI-optimized
points (`optimize_acqf`, gradient-based continuous local search) plus
UNSGA3-evolved points — never from an LLM/geometry-authored proposer.
Motivated by the "AI-discovered tuning laws" bioRxiv preprint (Tilbury et
al. 2025) — whose cleanest transferable lesson is role separation, e.g. a
parameter-initializer component explicitly banned from calling a solver —
this gate tested a **geometry-only** proposer (no GP/acquisition access at
all, mirroring that solver ban): three hand-designed strategies
(`gen_lhs_unexplored`: pure coverage/max-min-distance; `gen_perturb_front_
extremes`: local exploitation around Pareto-front x-locations, shrinking
radius; `gen_half_exploit_half_explore`: explicit batch split between the
two), fixed scoring (`trust_only`) and fixed top-k selection throughout, so
only generation varies — compared against a fixed-generation control
(`trust_only_topk`, i.e. the existing qLogNEHVI+UNSGA3 pool) and against
baseline, mAb, 3 replicates.

**Result**: no variant beat the control. `gen_lhs_unexplored` measurably
*lost* (mean −3.7% vs. control, 2/3 replicates individually significant
negative, consistent direction — the pure-coverage strategy, most divorced
from local exploitation, hurt the most). `gen_perturb_front_extremes` and
`gen_half_exploit_half_explore` were closer to a tie (means −3.1%, −1.7%
vs. control) but never individually significant, and mixed against
baseline. **Gate verdict: closed, negative.** Geometry alone — no gradient,
no posterior access — cannot replace `optimize_acqf`'s continuous local
search against the acquisition surface. This is also evidence *against*
expecting a hand/LLM-written geometric proposer to beat gradient-based
continuous optimization at its own game, which shaped how the later
compose_batch work (Part 7) was scoped.

## Part 5 — The batch-size ablation: does joint-batch *redundancy
accounting* explain the tie?

`trust_only_topk` vs. baseline, mAb, batch_size ∈ {1, 2, 3, 5, 10}, 3
replicates each. Motivation: qLogNEHVI collapses to plain per-point noisy
EHVI at q=1 — no batch to jointly reason about — so if the missing
ingredient were redundancy discounting (penalizing a candidate for being
correlated with another candidate in the *same* batch), the gap should
shrink or vanish as batch size drops toward 1.

**Result**: no trend. Mean diffs bounce around zero with no monotonic
pattern across batch size (bs=1: −1.2%, bs=2: +3.1%, bs=3: −2.7%, bs=5:
+2.6%, bs=10: +2.6%), zero significant results anywhere across all 15
replicate-level tests, and variance stays large and sign-flipping even
within one batch size. **Gate verdict: closed, negative — and a
refutation, not an inconclusive trend.** The tie exists uniformly across
*every* batch size tested, including q=1, where batch-redundancy
accounting is structurally irrelevant. This also corrects an earlier
speculation in this document (Part 3's original framing): LLaMEA-BO's
reported success in a sequential q=1 setting is **not** explained by
"redundancy accounting doesn't matter at q=1, so per-candidate methods
should work there" — our own q=1 comparison ties exactly like every other
batch size. Whatever separates LLaMEA-BO's setting from this project's
finding, it isn't batch-size-dependent redundancy discounting.

## Part 6 — The noise-integration gate: does MC integration over posterior
uncertainty explain the tie?

qLogNEHVI is *noisy* EHVI — it Monte-Carlo integrates over the joint
posterior even at q=1, weighing how trustworthy each point's estimated
Pareto-front position is given observation noise. `trust_only` and
`ehvi_approx` are point-estimate scorers — neither ever reads `std` in a
way that samples from it. `mc_hvi_approx` (numpy-only, sandboxed) draws
n=20 samples per candidate from its posterior `N(mean, std)` and averages
a per-sample HV-improvement proxy — a direct decomposition of the
"integrate over uncertainty" idea, deliberately without any joint-batch
reasoning (already ruled out by Part 5).

**Result**: null. `mc_hvi_approx_topk` vs. baseline: −0.2%/+2.4%/+0.6%
(mean +0.9%, never significant, inconsistent direction). Vs. `trust_only`:
mean −0.7%, inconsistent. Vs. `ehvi_approx`: mean −0.5%, one individually
significant *negative* result, inconsistent overall. **Gate verdict:
closed, negative.** Per-candidate MC integration over posterior uncertainty
does not close the gap either, on mAb.

## Part 7 — The composition gate, take two: does *genuinely joint* batch
construction explain the tie?

Parts 1–6 all held batch composition to "score independently, then select"
in some form. `compose_batch(context, k) -> list[int]` is a structurally
different contract: one sandboxed function that outputs the whole batch's
indices at once, with no intermediate per-candidate score — a direct
attack on decomposition itself, and the one lever LLaMEA-BO's own
sequential q=1 setting can't test either. Run as a pre-registered 2×2
ablation (surrogate: independent per-objective GPs vs. DA-COREG, a
coregionalized multi-task GP sharing structure across objectives ×
selection: baseline's own qLogNEHVI-score-plus-novelty-selection vs.
compose_batch), all four cells run together from the start (not
combined-then-backfilled) so a result is attributable to one lever, not an
artifact of only testing the combination — on synthetic ZDT1 first
(cheap, known-structure, shakes out new infrastructure before spending
real-domain compute), 3 replicates, 20 campaigns.

Seed 1, `greedy_marginal_hv`: sequential greedy hypervolume-improvement
batch construction over posterior means (the direct joint-composition
analog of qEHVI's own greedy construction, Monte-Carlo hypervolume
estimated over a shared sample grid).

**Result**: `compose_indep` (compose_batch + independent GPs) **lost**
significantly and consistently — mean −0.8%, all 3 replicates individually
significant negative (p=0.0005, 0.0064, 0.0094), 3/3 consistent direction.
`baseline_da_coreg` (DA-COREG alone, no compose_batch) tied cleanly (mean
+0.1%, never significant) — the surrogate swap alone is inert.
`compose_da_coreg` (combined) lost similarly to `compose_indep` alone
(mean −0.6%, 2/3 significant) — the loss tracks the compose_batch lever,
not the DA-COREG lever, resolving the ablation's own pre-registered
decision table cleanly.

Seed 2, `greedy_marginal_hv_mc`: identical greedy construction, but each
candidate's marginal HV contribution is averaged over MC samples from its
own posterior instead of using the raw mean — testing whether the loss
was specifically attributable to being uncertainty-blind (an
uncommitted-early, unreliable-mean-estimate overcommitment failure mode
that would predict this fix should help).

**Result**: no recovery. `compose_mc_indep` (mean −0.8%, 2/3 significant
negative) is statistically indistinguishable from mean-only `compose_indep`
(mean −0.8%). `compose_mc_da_coreg` (mean −0.5%, 2/3 significant negative)
similarly tracks `compose_da_coreg`. **Gate verdict: closed, negative —
and this closes it with more information than a tie would have.** The
uncertainty-blindness hypothesis is falsified directly: adding it changed
nothing. What's left as the explanation is more structural than a missing
term: greedy sequential construction with no backtracking (once a
candidate is accepted it is never reconsidered, unlike `optimize_acqf`'s
joint continuous relaxation over all q points simultaneously), independent
per-candidate MC sampling rather than one coherent joint sample across the
whole batch (real noisy-EHVI integrates over joint, correlated batch
outcomes, not marginal per-point ones), and a comparatively crude
500–800-point Monte Carlo hypervolume estimator. Continuing to patch this
family of heuristic would mean re-deriving qEHVI/qNEHVI's actual machinery
by hand — which stops being a test of "can decomposition beat joint
optimization" and becomes reimplementing the joint optimization badly.
**This algorithmic family was not carried to coatings or mAb** — the
synthetic gate's purpose (fail cheap before spending real-domain compute)
was served exactly as intended.

## Part 8 — DA-COREG alone: does surrogate structure, decoupled from
compose_batch, explain anything?

Part 7's ablation already isolated DA-COREG from compose_batch's own
failure (`baseline_da_coreg` tied cleanly on ZDT1, +0.1%, never
significant) — this promoted DA-COREG from "equal billing with
compose_batch" to the more clearly differentiated remaining direction,
tested entirely on its own: `strategy_ablation_cell(use_da_coreg=True,
use_compose=False)` vs. `mo_egbo_novelty`, no compose_batch anywhere.

**Prerequisite check, before spending compute**: DA-COREG's entire value
proposition is sharing structure across *correlated* objectives — a
coregionalized multi-task GP has nothing to exploit if the objectives are
independent. Direct Pearson correlation check across every domain in this
project: coatings' two objectives are confirmed near-independent
(r=+0.0032 — this was already known to be the deliberate design of that
domain's objective pair, now directly reconfirmed), while **mAb was never
previously checked and turns out to have real structure** (Tm–kD
r=+0.60, Tm–viscosity r=+0.23) — comparable in magnitude to DTLZ2's
strongest engineered pair (r=−0.56). This reprioritized the real-domain
test order: DTLZ2 (positive control) → mAb (highest real-domain prior) →
coatings (low prior, confirmed).

**DTLZ2 positive-control result**: `baseline_da_coreg` **lost**,
consistently and worsening — mean −2.0% across 3 replicates (−0.8%, −1.5%,
−3.8%), 0/3 replicates positive, one severely significant loss (0/20 wins,
p≈1.9×10⁻⁶ — the theoretical minimum achievable two-sided Wilcoxon p at
n=20, not a display rounding artifact). This is the positive-control gate
failing in the losing direction, on the one domain engineered to have real
cross-objective correlation for DA-COREG to exploit — exactly the result
the gate exists to catch before real-domain compute is spent, and it did.

**Diagnosing the loss, three checks, in order of cost**:
1. *Silent fallback* (the mechanism behind two earlier bugs in this
   project — the `numpy.random` sandbox issue and a silently-swallowed
   `KeyError`): re-probed the exact 0/20 replicate's seed block directly.
   Zero fallback batches out of 120 — `strategy_ablation_cell` never once
   caught an exception. Ruled out.
2. *Column-ordering / output-shape bug* (the same category as this
   project's own earlier hardcoded-objective-direction bug): a synthetic
   sanity check fit DA-COREG on three trivially-distinguishable per-task
   targets (offsets of 0/100/200) and confirmed the posterior's output
   columns correspond to the correct objectives. Ruled out.
3. *Poor surrogate fit from a data-starved coregionalized MLE* (the
   leading hypothesis going in — more hyperparameters, same ~10-40 point
   budget as every independent-GP fit elsewhere in this project):
   held-out Gaussian NLL/RMSE, DA-COREG vs. independent GPs, fit on the
   exact same reconstructed campaigns, evaluated against other pool
   points' true noiseless DTLZ2 values. **Falsified — DA-COREG's fit was
   actually *better*** (mean held-out NLL −0.375 vs. independent GPs'
   +0.215, lower RMSE in most objective/campaign combinations). The
   surrogate itself is not the problem.

**What the evidence instead points to**: instrumenting the exact
acquisition code path `strategy_ablation_cell` uses (both the per-candidate
`acq_fn` evaluation loop and `optimize_acqf`) showed zero failures for
either surrogate — but a striking difference in acquisition-VALUE spread
specifically in the two campaigns where independent GPs found the sharpest
candidate differentiation (std≈7–8): DA-COREG's spread was 3–4× tighter in
those same campaigns (std≈1.8–2.1), while comparable in the other three.
The plausible mechanism: `qLogNoisyExpectedHypervolumeImprovement` draws
joint Monte Carlo samples across objectives to estimate expected
hypervolume improvement; a `MultiTaskGP` that has correctly learned strong
cross-objective correlation (confirmed good by the NLL check above) will
produce MC samples that move together across objectives rather than
independently, which can compress the acquisition value's dynamic range
across candidates even when marginal predictions are individually
accurate — good calibration, worse discrimination between candidates once
integrated into the batch acquisition value. A secondary, confirmed-but-
likely-not-causal observation: botorch silently downgrades
`cache_root=True` to `False` for any `MultiTaskGP` (a logged
`RuntimeWarning`, not an error) — `qLogNEHVI` was not built with
`MultiTaskGP` as a first-class input, which lends outside plausibility to
"this pairing has rough edges" as a category, though this specific flag
is a compute-caching optimization, not something that should change
acquisition values, and no failures were observed consistent with it
mattering here.

**Status: three concrete bug hypotheses ruled out with direct evidence
(fallback, column-ordering, fit quality); one plausible, data-consistent,
not-fully-proven structural mechanism identified (correlated joint MC
sampling compressing acquisition-value discrimination) — not chased to
full certainty, since doing so (inspecting raw MC sample tensors inside
qLogNEHVI) is a larger investment than this direction's own demoted
priority warrants.** DA-COREG in its current form was not carried to mAb
or coatings — the positive-control gate did its job.

## What this motivates

Eight independent mechanism probes now converge on the same conclusion
from every angle tested: per-candidate scoring quality (Parts 1–2,
vocabulary/adaptivity), batch selection method (Part 3), candidate
generation (Part 4), batch-size/redundancy accounting (Part 5),
per-candidate noise integration (Part 6), genuinely joint batch
composition — both mean-only and uncertainty-aware (Part 7) — and
coregionalized surrogate structure (Part 8). None of them, individually or
in the combinations tested, closes the gap to `mo_egbo_novelty`. Two of
the eight (compose_batch and DA-COREG) didn't just fail to help — they
actively hurt, significantly and consistently, when tested on a positive-
control domain engineered to favor them. The design rule this body of
evidence supports, now on firmer footing than any single result could give
it: **qLogNEHVI's joint, MC-integrated, gradient-optimized treatment of the
whole batch is doing work that no decomposition or component substitution
tested here — by scoring, by selection, by generation, by an alternative
form of joint composition, or by a richer surrogate — recovers.** A
campaign planner should not expect to replace a jointly-optimized batch
acquisition function with a simpler decomposed alternative or a modified
surrogate, however that alternative is chosen, hand-designed, or evolved —
and should be specifically wary of components that look correct in
isolation (a good surrogate fit, a sound joint-composition algorithm) but
interact poorly with the rest of an already-tightly-integrated pipeline.

This also closes out "approach K" as originally framed. LLaMEA-BO already
demonstrates "LLM rewrites the full BO algorithm" in a sequential q=1
setting, undercutting that framing as a standalone novelty claim; the
specific corner its own setting couldn't test — within-batch joint
composition — is now tested here across two designs and closed negative,
and Part 5 directly rules out the specific hypothesis ("redundancy
accounting doesn't apply at q=1") that would have reconciled LLaMEA-BO's
reported success with this project's ties. Whatever explains that
difference, it is not batch-size-dependent, and it is not fixed by any
decomposition- or surrogate-based lever tested in this document. The
remaining honest open question is not "which decomposition works" — eight
have now been tried — but why coatings shows a modest, real, replicated
edge for exploitation-flavored AFs at all when mAb shows none; that
mechanism (Part 2's "what is explicitly not claimed" section) remains
unresolved and is the one genuinely open thread this investigation leaves
behind.
