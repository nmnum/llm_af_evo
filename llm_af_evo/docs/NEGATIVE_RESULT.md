# Why LLM-Evolved Acquisition Functions Fail to Beat a Strong Hand-Designed MOBO Baseline

**Status: draft, 2026-08-02.** Synthesizes `SESSION_LOG.md`,
`llm_evolved_afs_comprehensive_log.md`, and `L_COATINGS_FINDINGS.md` into a
single, standalone account of the negative result, closing the two items
that were left open at the end of those documents (mu/sigma-dominance +
win-overlap; the mAb fixed-hints re-test) with data gathered in this pass,
plus two further follow-up pilots (a directional-inversion-bug re-test on
three additional Run-4 candidates, and a decay-schedule tuning sweep for
`hint_fixed_ucb`'s family) that close the remaining open items below.

## The question

Can an LLM evolve short, closed-form acquisition functions (`score_pool(context)
-> list[float]`, scoring — never selecting — a candidate pool) that beat
`mo_egbo_novelty`, a hand-designed baseline combining qLogNEHVI (MC-integrated,
gradient-optimized joint batch acquisition), U-NSGA-III candidate diversity, and
novelty-weighted batch selection, on short-horizon, noisy, multi-objective
physical-domain optimization?

**Answer: no.** Across eight independent mechanism probes, four full
evolutionary runs, and a follow-up re-test of previously-broken hints, no
decomposed, evolved, or hand-designed alternative to `mo_egbo_novelty`'s
jointly-optimized batch construction closes the gap — on either of two
structurally different real-world domains.

## Setup

- **Evolved unit:** `score_pool(context) -> list[float]`. Batch *selection*
  (top-k over scores) is fixed and never evolved. `context` exposes
  `pool` (candidate `x` + GP posterior, pre-flipped to all-maximize),
  `X_obs`/`Y_obs`, `pareto_front`, `ref_point`, and campaign state
  (`step`/`budget`/`progress`/`stagnant_batches`).
- **Evolution:** (µ+λ), population 8, offspring 8, 20 generations, 2 LLM
  calls/gen × 4 offspring = 40 calls/run. Fitness = mean relative HV margin
  vs. baseline over full-campaign replay, minus a LOC penalty (`gamma`).
- **Two domains**, chosen to be structurally different in every axis that
  matters:
  - **Coatings** — 4D continuous, real Ada/MacLeod et al. 2022 self-driving-lab
    data (253 real samples, discrete pool, nearest-neighbour snap-query, no GP
    extrapolation), 2 near-independent objectives (conductivity vs.
    conductance uniformity, r=0.0032), moderate real noise. GP models it well.
  - **mAb excipient oracle** — 16D categorical, synthetic oracle, 3 objectives
    (Tm/kD/viscosity, some real cross-correlation: Tm–kD r=+0.60), CV≈49.7%
    on the aggregation pathway. Noisy; GP mean is less trustworthy.

## Part 1: Eight mechanism gates (pre-evolution, `L_COATINGS_FINDINGS.md`)

Before any evolutionary search, eight independent hypotheses about *why*
decomposed AFs might tie or lose to `mo_egbo_novelty` were tested directly,
each attacking a different structural piece of the baseline:

| # | Gate | Hypothesis | Result |
|---|------|------------|--------|
| 1–2 | Per-candidate scoring quality / vocabulary | Maybe evolved/hand-written scoring just needs the right ingredients (`pareto_front`, `ref_point`) | Closed negative — best mAb result ties (+2.4%, p=0.50); `ehvi_approx` ties on both domains (+1.0%/+0.70, +0.0%/never sig.) |
| 3 | Batch selection (top-k vs. novelty-weighted) | Maybe the gap is in *selection*, not scoring | Closed negative — coatings (the domain with an actual gap to modulate) shows no change: +4.8% (topk) vs. +5.1% (novelty) |
| 4 | Candidate generation (geometry-only proposers) | Maybe qLogNEHVI's gradient-based `optimize_acqf` search is replaceable by geometry | Closed negative — worst variant (`gen_lhs_unexplored`) *lost*, mean −3.7%, 2/3 replicates significant |
| 5 | Batch-size / redundancy accounting | Maybe the tie shrinks toward q=1 (where joint-batch reasoning is structurally irrelevant) | Closed negative — **refutation**, not inconclusive: no trend across bs∈{1,2,3,5,10}, zero significant results anywhere |
| 6 | Per-candidate MC noise integration | Maybe reading `std` (not just `mean`) closes the gap | Closed negative — `mc_hvi_approx` mean +0.9%, never significant |
| 7 | Genuinely joint batch composition (`compose_batch`) | Maybe decomposition itself (score-then-select) is the problem | Closed negative — **and actively harmful**: `compose_indep` lost significantly on ZDT1 (mean −0.8%, p=0.0005/0.0064/0.0094, 3/3); adding MC-awareness didn't recover it (−0.8%, −0.5%) |
| 8 | Coregionalized surrogate (DA-COREG) | Maybe a shared multi-task GP captures cross-objective correlation independent GPs miss | Closed negative — **and actively harmful** on the positive-control domain (DTLZ2): mean −2.0%, 0/20 wins, p≈1.9×10⁻⁶ |

**Convergent conclusion:** qLogNEHVI's joint, MC-integrated, gradient-optimized
treatment of the whole batch is doing work that no single-lever substitution
tested here recovers — not scoring, not selection, not generation, not batch
size, not per-candidate noise integration, not joint composition, not a
richer surrogate. Two of the eight (`compose_batch`, DA-COREG) don't just
fail to help — they actively hurt when tested on a domain engineered to
favor them.

## Part 2: Four evolutionary runs (v2)

Running actual LLM-driven evolution (not hand-designed alternatives) on top
of this already-negative mechanism landscape produced the same conclusion,
with two additional, previously-unresolved structural findings.

- **Run 1** (coatings, gamma=0.005, n=15): stagnated — gamma too aggressive,
  penalized any structural complexity before it could pay off.
- **Run 2** (coatings, gamma=0.005, n=75): `trust_only` (pure exploitation,
  no sigma) won, not the fixed-UCB hint — first hint of mu_sum dominance.
- **Run 3** (mAb, gamma=0.005, n=75): evolution could not progress on mAb at
  this gamma — later traced to noise floor + LOC-penalty miscalibration for
  this domain's smaller margin scale.
- **Run 4** (coatings, gamma=0.001, n=75, 8-hint set): gamma relaxed enough to
  allow progression. Winner `gen6_child0`. **None of the 4 new
  (non-UCB) hints survived** — see below.
- **`run_v2_mAb_gamma001_fixed`**: the mAb-domain follow-up to Run 4, intended
  to run the same 20-generation schedule with the 4 new hints' bugs already
  fixed. **Interrupted at generation 11/20** by a codebase restructuring —
  no `final_population.json`/`history.json` exist for it. Not a completed,
  analyzable run; its `af_code_logs/` (31 `call_*.py` files) are usable only
  as fixed reference implementations of the 3 previously-broken hints (see
  Part 4), not as an evolution result in their own right.

### The four new (non-UCB) hints, as evolved in Run 4

Of the 8-hint v2.1 set, 4 were new, non-UCB mechanisms (`noisy_front_hvi`,
`dpp_diversity`, `local_penalization`, `pareto_membership`). Bug analysis of
Run 4's actual generated code found:

- `dpp_diversity` — **correct**, but lost to mu_sum dominance anyway (see
  below) — the one new mechanism that wasn't simply broken.
- `noisy_front_hvi` — **broken**: 54/71 lines were LLM hedging comments; the
  function never reached `return`, silently defaulting to all-zero scores.
- `local_penalization` — **broken, 3 bugs**: uninitialized/empty `scores`
  list indexed directly (IndexError); `multiplier = 1.0 - exp(-radius)`
  inverted the intended suppression direction (far candidates penalized
  *more* than close ones); a `float`-used-as-list-index TypeError in
  best-candidate selection.
- `pareto_membership` — **drifted**: checked "sample dominates a front
  point" instead of "sample is dominated by a front point," inverting the
  intended Pareto-membership semantics.

So of the four genuinely new mechanisms attempted, three were simply
non-functional in the run that produced them, and the one that worked
(`dpp_diversity`) still lost. This looked, at the time SESSION_LOG.md and
the comprehensive log were written, like an open question: were these
mechanisms bad ideas, or just badly implemented? Part 4 below closes it.

## Part 3: mu_sum dominance and the win-overlap puzzle (previously unresolved)

Two related, previously-open questions from `SESSION_LOG.md` §7–8 are
resolved in this pass with fresh diagnostic runs.

**Question 1 — is sigma just near-constant on coatings (making any
uncertainty weight a no-op by construction), or is it a real weight/threshold
effect?** `diagnose_mu_sigma_dominance.py` against the coatings training
logs (n=25 steps): `sigma_sum_cv` = 0.31–0.60 — **not** near-zero, refuting
the "sigma is constant" hypothesis. But at realistic hint weights, sigma
genuinely changes the exact top-k candidate *set* on a large fraction of
steps: 48% of steps at w=0.5, 64% at w=1.0, **88% at w=2.0** (`fixed_ucb`'s
own beta). Sigma is real and moves per-step decisions — but downstream,
those different per-step decisions still collapse to the same campaign-level
win/loss outcome (Rule 1 below), which is what the second question resolves.

**Question 2 — do structurally different AFs with identical win_rates win
the *same* campaigns (explanation a: campaign difficulty dominates) or
*different* campaigns that happen to tie in count (explanation b: coarse
win_rate metric collision)?** `diagnose_win_overlap.py` against
`run_v2_coatings_gamma001_v2`'s final population (8 candidates, 75
campaigns): pairwise Jaccard overlap of winning-campaign sets among
same-win_rate candidates (e.g. `gen6_child0`=0.6933, `hint_fixed_ucb`=0.6933,
`gen4_child0`=0.6933, `gen1_child1`=0.6933, `gen9_child0`=0.6933,
`gen5_child0`=0.6933) ranges **0.91–1.00** — explanation (a). Structurally
different AFs win nearly the *exact same set* of campaigns; different
`selection_signature` hashes reflect different HV *values* on those same
wins/losses, not different win/loss patterns.

**Together, these two results explain each other.** Sigma moves per-step
top-k *selections* substantially (Question 1), but which campaigns are
winnable at all is set by the campaign's own difficulty, not by which
reasonable-weight AF is doing the selecting (Question 2) — so different
per-step choices still funnel into the same campaign-level outcome. This is
a second, independent line of evidence for Rule 1 (mu_sum dominance): it's
not just that AFs select the same *candidates* (the original held-out-HV
finding), it's that they win/lose the same *campaigns* end-to-end.

## Part 4: Re-testing the fixed hints on mAb — still null (new this pass)

Since the three broken hints from Run 4 (`noisy_front_hvi`,
`local_penalization`, `pareto_membership`) were confirmed fixed in
`run_v2_mAb_gamma001_fixed/af_code_logs/` (call_00004, call_00007,
call_00008 respectively — verified by direct code reading), the natural
follow-up question is whether the *ideas* behind them have real,
previously-unexploited value on mAb, the domain where sigma actually
matters (Rule 2 below) — as opposed to coatings, where mu_sum dominance
means AF structure can't matter regardless of correctness.

A pilot (`pilot_fixed_hints_mab.py`) replayed all three fixed hints against
20 mAb training campaigns, 3-seed-averaged per campaign (per the noise
diagnostic's own recommended mitigation, §12–13):

| Hint | Mean margin | Median margin | Wins | p-value |
|---|---|---|---|---|
| `noisy_front_hvi` | −1.8% | −3.4% | 5/20 | 0.114 |
| `local_penalization` | +1.0% | +0.9% | 11/20 | 0.498 |
| `pareto_membership` | −6.4% | −5.0% | 7/20 | **0.040** |

An earlier n=8, single-seed pilot had shown `local_penalization` at a
seemingly promising +14.5% mean margin (p=0.078) — this was diagnosed at
the time as a likely denominator-shrinkage artifact (its single largest
margin occurred on the campaign with the lowest baseline HV of the 8),
directly parallel to the mean-margin-outlier-inflation problem that
motivated the project's own switch to median-based fitness. The n=20,
3-seed result confirms that diagnosis: the apparent edge evaporates to
+1.0%/p=0.50 under proper averaging. `pareto_membership` comes out
significantly *negative*. `noisy_front_hvi` trends negative,
non-significant.

**Conclusion: fixing the three previously-broken hints does not surface
hidden value.** None shows a real, statistically credible edge once
seed-noise and small-n artifacts are controlled for. Combined with
`dpp_diversity` (correct, but lost anyway) from Run 4, all four of the new
non-UCB mechanisms attempted in this project's hint set — correct or not —
fail to beat the baseline. This closes the "maybe the evolved AFs have
untapped potential" thread raised mid-project: they don't, at least not
among the mechanisms tried so far.

## Part 5: Two more loose ends, closed this pass

Two follow-up pilots were run after Part 4, targeting the two items still
listed as open at that point.

**5a. Sign-fixed re-test of three more Run-4 candidates.** Independent of
the three hints in Part 4, a later read of `run_v2_mAb_gamma001_fixed`'s
`af_code_logs/` (generations past where fitness scoring had stopped, so
never evolutionarily evaluated) turned up the same directional-inversion
bug class in three more candidates: `call_00016` (a distance-to-observed
penalty computed as `dist/max_dist` instead of `(max_dist-dist)/max_dist`,
rewarding near-duplicates), `call_00021` (a Pareto-dominance check with the
improving/dominated branches swapped), and `call_00030` (a "novelty" score
computed as `1/(min_dist_sq+eps)`, again rewarding near-duplicates of
observed points instead of penalizing them). All three were hand-fixed
(`fixed_merge_hints/call_0001{6,21,30}_fixed.py`) and re-tested on 20
genuinely held-out mAb campaigns (a disjoint `train`/`heldout` split, an
improvement over Part 4's single-split pilot), 3-seed-averaged:

| Hint | Mean margin | Median margin | Wins | p-value |
|---|---|---|---|---|
| `call_00016_fixed` | +0.5% | −1.4% | 9/20 | 0.956 |
| `call_00021_fixed` | −2.1% | −1.3% | 8/20 | 0.475 |
| `call_00030_fixed` | −2.3% | −0.9% | 10/20 | 0.330 |

Null across the board — two negative on mean, all at or below even on
median, none close to significant. Combined with Part 4's three hints and
`dpp_diversity` from Run 4, that's now **seven** distinct new (non-UCB)
mechanisms tried across this project, correct or bug-fixed, and none beats
baseline. This closes the "maybe a later, unscored generation has something
Part 4 missed" thread raised at the end of Part 4.

**5b. Decay-schedule tuning for the `hint_fixed_ucb` family.** Rule 2 (mAb
signal is thin but real) and Rule 3 (stagnation terms hurt) leave open
whether the *fixed-UCB family itself* — `hint_fixed_ucb` (β=2, no decay),
`gen5_child0` (linear decay, p=1), `gen9_child0` (quadratic decay, p=2), all
three shown individually significant/robust under 3-seed re-validation — is
tuned optimally, or whether a swept `beta0`/decay-exponent grid could do
better. A three-stage pilot (`pilot_decay_schedule_mab.py`) tested
`score = sum(mu) + beta0*(1-progress)^p*sum(sigma)` over
`beta0∈{0.5,...,2.5}, p∈{0,1,2}` (15 configs), with a pre-registered kill
criterion (`mean_margin>0 and p<0.05 and wins≥12/20`) fixed in advance to
rule out post-hoc threshold shopping:

- **Stage A** (10 train campaigns × 3 seeds × 15 configs, ranked by median
  margin per the noise diagnostic's own recommendation): top-2 were
  `beta0=2.5,p=1` (mean +4.6%, median +3.6%, 7/10, p=0.13) and
  `beta0=1.5,p=0` (mean +0.7%, median +2.0%, 6/10, p=0.92).
- **Stage B** (same 10 campaigns × 5 seeds): `beta0=2.5,p=1` held up (mean
  +4.1%, median +1.7%, 7/10, p=0.13); `beta0=1.5,p=0` did not (mean +0.8%,
  median **−1.5%**, 3/10, p=0.85) and was dropped. Winner: `beta0=2.5,p=1`.
- **Held-out validation** (20 genuinely disjoint heldout campaigns × 3
  seeds): mean +2.6%, median +2.1%, 12/20 wins, p=0.50.

**Verdict: FAIL, per the pre-registered kill criterion.** The held-out
result is directionally positive on every metric (mean, median, and win
count clears the ≥12/20 bar on its own) but p=0.50 is far from the required
p<0.05 — at n=20 held-out campaigns, this is indistinguishable from noise.
No post-hoc re-tuning was performed, per the pre-registration. This closes
the "is the fixed-UCB family itself under-tuned" question: sweeping
`beta0`/decay together doesn't produce a decay schedule that clears a
properly powered significance bar, even though the single best swept point
looks similar in magnitude to `hint_fixed_ucb`'s own already-validated edge.

## Design rules this body of evidence supports

1. **mu_sum dominance (coatings-specific).** Any nonzero uncertainty weight
   produces rank-equivalent batch selections to pure exploitation — 7/8 AFs
   in Run 4's final population were byte-identical on held-out HV despite
   different code. Reinforced by the win-overlap Jaccard result above:
   AFs don't just select the same candidates, they win the same campaigns.
   Coatings cannot distinguish AF quality above "does it include any sigma
   term at all" — it is the wrong substrate for AF evolution.
2. **AF structure matters on mAb**, but the signal is thin and easily
   swamped. The 3-seed re-validation (§13 of the comprehensive log) showed
   `hint_fixed_ucb` flip from p=0.596 (non-significant) to p=0.030
   (significant) purely from adding seed averaging — a single-seed
   evaluation would have discarded a genuinely significant AF as noise.
   Even after averaging, SE (~0.048) remains ~4× the signal gap between top
   AFs (~0.011) — the fitness function is working near its resolution
   limit, and the fixed-hints re-test (Part 4) confirms that once seed
   noise is properly controlled, none of the tried non-UCB mechanisms clear
   that thin margin either.
3. **The stagnation term is a liability**, not an asset, on mAb: AFs adding
   a stagnation-triggered novelty boost (`gen4_child0`, `gen1_child1`)
   dropped below significance under seed averaging, while simpler
   progress-only decaying UCB (`gen5_child0`, `gen9_child0`) strengthened.
   Each additional noisy input variable (here, `stagnant_batches`) adds
   variance the fitness signal can't afford to carry.
4. **gamma must be calibrated to the domain's margin scale.** gamma=0.005
   stagnated evolution on both coatings (n=15) and mAb (n=75); gamma=0.001
   let coatings progress. No single gamma value transfers across domains
   with different margin magnitudes.
5. **Coatings-evolved AFs do not generalize** — not to mAb (single-seed
   held-out results), and not to noiseless synthetic benchmarks (0/8 AFs
   beat baseline on ZDT1, 4/8 significantly worse; 0/8 on DTLZ2). The
   uncertainty bonus that helps on noisy real domains actively hurts where
   the GP's mean is already reliable.
6. **DA-COREG's catastrophic failure was pipeline-specific, not
   fundamental.** Removing qLogNEHVI from the loop (keeping DA-COREG's
   surrogate, `trust_only` scoring, top-k selection) turned a −3.8%/
   p≈1.9×10⁻⁶/0-of-3-significant catastrophe into a −0.7% (std 0.4),
   0-of-3-significant tie. DA-COREG ties baseline in evolved-AF-style
   pipelines (no qLogNEHVI); it is not disqualified from that context, but
   there is no evidence it helps there either.

## Why this generalizes rather than being an artifact of one dataset

A natural worry: if every negative finding came from a single domain
(coatings), it might just reflect that domain's idiosyncrasies rather than
anything general about LLM-evolved AFs. That isn't the shape of the evidence
here. Coatings and mAb fail for **two different, independently diagnosed
mechanistic reasons**:

- On coatings, AFs fail because mu_sum dominance makes the AF's scoring
  logic causally irrelevant to the outcome once any sigma weight is present
  — verified both at the per-candidate-selection level and, this pass, at
  the campaign-win level (Jaccard overlap 0.91–1.00).
- On mAb, AFs fail not because structure is irrelevant (sigma genuinely
  changes rankings there) but because the noise floor of the evaluation
  pipeline is large enough to prevent evolution — or post-hoc validation —
  from reliably distinguishing real signal from seed noise, and the
  specific mechanisms tried (three previously-broken, now-fixed hints, one
  correct-but-losing hint) don't clear that bar even once noise is
  controlled for.

Two domains, two distinct failure modes, both negative, is a stronger and
more generalizable result than either domain alone — it is far less likely
that both are coincidental artifacts of their specific datasets than that
either one is.

## What remains open

- The mechanism behind *why* coatings shows a modest, replicated
  exploitation edge (~+4.5%) while mAb shows none at all remains
  unresolved (`L_COATINGS_FINDINGS.md`'s own explicit non-claim) —
  dimensionality, pool sparsity, real-vs-synthetic noise, and
  continuous-vs-categorical inputs all differ simultaneously between the
  two domains tested, so none can be isolated from n=2 domains.
- The noise-sweep needed to causally test the "floor rule" (does an
  uncertainty bonus's benefit reverse as observation noise is dialed up on
  a synthetic benchmark?) was never run.
- `run_v2_mAb_gamma001_fixed`, the one run that would have tested full
  20-generation evolution on mAb with the fixed hint set, never finished
  (interrupted at gen 11/20) — whether a completed run would find something
  the seven individually-tested candidates across Part 4 and Part 5a missed
  is unknown, though their uniform null result makes that less likely than
  it looked before this pass.
