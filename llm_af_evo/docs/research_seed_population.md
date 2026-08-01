# Research note: how prior LLM/evolutionary program-search systems seed their initial population

## Scope and method

Primary sources consulted directly (arXiv HTML/PDF, Nature/DeepMind materials),
not blog summaries, except where noted. Focused on: what seed program(s) each
system starts from, whether that's diverse hand-written baselines vs. a single
trivial seed vs. random init, and any stated claim linking seed choice to
convergence speed or search-space exploration.

## 1. FunBO — Aglietti et al., "FunBO: Discovering Acquisition Functions for
Bayesian Optimization with FunSearch" (arXiv:2406.04824, 2024; also at
OpenReview id=XjbJR9374o)

This is the most directly relevant prior system — `evolve_af.py`'s own
docstring names it explicitly ("FunBO-style programs-database evolution").

- **Seeding**: FunBO uses a **single hand-written initial acquisition
  function** `h`, not a diverse set. Per the paper: *"FunBO's initial program
  h determines the input variables that can be used to generate alternative
  AFs while imposing a prior on the programs the LLM will generate at
  successive steps."* The initial program's functional form is built around
  Expected Improvement (EI), but exposes EI/UCB/POI-style inputs (posterior
  mean, posterior variance, incumbent value, a beta-style parameter) so the
  LLM can recombine them.
- **Rationale (paraphrased from the paper)**: the seed's role is not "give the
  search several strong starting points to choose among" but "declare which
  variables and functional vocabulary are legal," i.e. it constrains the
  *input schema* the LLM is allowed to build with, more than it diversifies
  starting strategies.
- **Islands, not diverse seeds**: FunBO's programs database is initialized
  with `N_db = 10` islands that evolve independently, but **every island is
  seeded with a copy of the same single initial program** — diversity comes
  from independent evolutionary trajectories across islands (and LLM
  stochasticity), not from different hand-written starting strategies.
- **No stated claim about seed diversity vs. convergence.** The paper does
  not analyze or claim anything about how the choice of seed program, or
  seeding with one vs. many baselines, affects convergence speed or breadth
  of the explored program space. This is a genuine gap in FunBO's own
  reporting, not something I'm inferring against the source.
- **"EDGAR" / graphical diagnostics**: I could not find any mention of
  "EDGAR," a "Harris group," or graphical/plot-based diagnostics shown to the
  LLM anywhere in the FunBO paper (full-text search of the HTML version came
  back empty on all three terms). See the "Harris group / EDGAR" note below —
  I was unable to substantiate this as a real primary source at all.

## 2. "The Harris group's EDGAR system" — **now verified: real, wrong domain searched**

**Correction to my earlier search**: this is a real primary source. My
searches above looked for it in the chemistry/materials BO literature under
the literal name "EDGAR" and came up empty — the paper is actually in
**neuroscience**, and "EDGAR" doesn't appear in it as a system name at all
(that label may have been introduced elsewhere, or misremembered; the
Harris-group attribution is correct). The paper, supplied directly and read
in full:

**Tilbury, Kwon, Haydaroglu, Ratliff, Schmutz, Carandini, Miller, Stachenfeld,
Harris. "Characterizing neuronal population geometry with AI equation
discovery." bioRxiv, doi:10.1101/2025.11.12.688086 (posted 2025-11-13).**
Kenneth D. Harris (UCL) is corresponding author — confirming the "Harris
group" attribution `evolve_af.py`'s docstring makes.

This system evolves Python programs representing **neuronal tuning-curve
equations** (not acquisition functions — the "AF" domain match to this repo
is structural/methodological, not literal) via LLM-driven programs-database
evolution, closely paralleling FunSearch/AlphaEvolve's mechanics but adding
two features neither of those has: (1) **graphical diagnostics** (scatter
plots of parent-model fits vs. raw data for 9 cells) embedded directly in
the prompt alongside code, and (2) **required docstrings** on every
LLM-generated program, used both to convey the LLM's own reasoning back to
the reader and (per the paper) referencing specific subplot evidence from
the graphical diagnostics.

**Seeding, directly answering this research question**: "Each island is
initialized with **two seed programs**: Single-peaked Gaussian... Double-
peaked Gaussian... Both programs include simple parameter estimators using
circular statistics and peak detection." This is the real, load-bearing
data point for this repo's seed-population question — and it lands
**between** this repo's 7-diverse-seed design and FunBO/FunSearch/
AlphaEvolve's single-seed pattern:
- Not one seed (unlike FunBO/FunSearch/AlphaEvolve).
- Not seven, either — **two**, and both are close variants of the *same*
  functional family (Gaussian bumps), differing only in whether there's one
  peak or two. Neither seed exercises a fundamentally different *strategy*
  the way this repo's trust_only/novelty_only/ehvi_approx do — they exercise
  a fundamentally different *shape count*, and nothing else.
- Seeds are replicated identically across all `N_islands = 8` islands — same
  "one seed set, many islands, diversity from independent trajectories"
  pattern as FunBO/FunSearch, just with a 2-program set instead of a
  1-program set.
- **No stated rationale for why two, or why these two specifically** —
  the paper doesn't argue that two Gaussians was chosen for coverage of a
  strategy space; it reads as "the two most natural starting points for
  *this specific domain*" (single vs. double-peaked neurons are a known,
  pre-existing distinction in the tuning-curve literature — see the paper's
  own citation of the classic "double Gaussian" model, ref. 11), not a
  deliberately constructed diversity set.
- Confirms this repo's separately-added **docstring requirement is directly
  precedented**: the paper states plainly that graphical-diagnostic feedback
  "also provides insight into the reasoning the LLM used when crafting the
  models, which can be found in the docstrings of the LLM-generated code"
  (Methods, "Multi-modal Prompt Engineering" section) — i.e. the Harris
  group's own system independently converged on exactly the same
  "docstring = interpretability trace for a human reading the evolution's
  history" argument that was the reasoning above for adding the
  explainability-line requirement to `evolve_af.py`. The example prompts in
  their Appendix 2 also show `neuron_model_v3`'s docstring explicitly
  narrating "which parent contributed what" (e.g. "parent_model_1:
  VariableExponent_AsymmetricBimodalModel... parent_model_2:
  IndependentLocation_AsymmetricPrimaryModel... This model creates a
  comprehensive bimodal tuning curve by intelligently combining...").
- Also confirms two other things this repo already does, independently
  arrived at: a **complexity penalty** on free-parameter count (their
  Extended Data Fig. 2 ablates λ=0 vs. λ>0 and shows parameter count grows
  unboundedly without it — directly analogous to this repo's `gamma * LOC`
  term and its own stated rationale), and **deduplication** of
  functionally-similar programs (their "cosine similarity > 0.99 on
  evaluation matrices and score difference < 0.025" rule is the same idea as
  this repo's `selection_signature` rank-equivalence dedup, just measured
  differently — behavioral fingerprint vs. exact-tie hash).

## 3. FunSearch — Romera-Paredes et al., "Mathematical discoveries from
program search with large language models," Nature 625, 2024 (also
DeepMind blog + google-deepmind/funsearch GitHub)

- **Seeding**: a **single trivial/simple hand-written program** per problem —
  e.g. for the cap set problem, a straightforward priority function that
  produces far-from-optimal cap sets on its own. FunSearch does not seed with
  multiple diverse strategies; it seeds with one deliberately weak starting
  point and relies on the evolutionary loop plus LLM mutation to discover
  strong variants from there (this is consistent with the well-known
  headline result: starting from a trivial priority function, FunSearch found
  one yielding a cap set of size 512 in 8 dimensions, beating the
  previously best-known construction of 496).
- **Islands**: the programs database is split into `m` independent islands,
  **each initialized with a copy of the same user-provided initial
  program** — same "one seed, replicated across islands, diversity emerges
  from independent evolution" pattern as FunBO (unsurprising, since FunBO is
  explicitly built on FunSearch's machinery).
- **Stated rationale**: FunSearch's own framing is that the initial program
  need only be *correct*, not *good* — the system's value proposition is
  precisely that weak/trivial starting points can be evolved into
  state-of-the-art ones. There is no claim in the source that diverse
  hand-written seeds would converge faster or explore more broadly than a
  single trivial one; if anything the paper's selling point argues the
  opposite (a single weak seed is sufficient).

## 4. AlphaEvolve — Novikov et al., "AlphaEvolve: A coding agent for
scientific and algorithmic discovery" (arXiv:2506.13131, DeepMind, 2025)

- **Seeding**: AlphaEvolve seeds with a **single human-written program**
  marked out via "evolve blocks" in the user's code — *"any user-provided code
  inside such evolution blocks serves as the initial solution to be improved
  by AlphaEvolve."* For open-ended math-construction problems, the seed is
  explicitly described as "a simple or a random construction," matching
  FunSearch's approach rather than diversifying it.
  - For engineering-heavy applications (e.g. the paper's kernel/scheduling
    work), the seed is instead **an existing production baseline
    implementation**, i.e. "start from what already works and evolve it,"
    not a set of alternative baseline families.
- **Note on human-injected diversity, not seed diversity**: the paper
  mentions that for some tasks, "seeding the initial program with our own
  ideas (such as adding stochasticity to the evaluation function or using
  evolutionary approaches) could further boost performance" — this is about
  enriching the single seed's content/ideas, not about running multiple
  distinct hand-written seed programs in parallel.
- **No isolated ablation of seeding strategy.** The paper's ablations
  (Section 4: evolution, context, meta-prompts, full-file evolution, model
  capability) do not include an ablation isolating the effect of seed choice
  or seed diversity on convergence speed or exploration breadth.
- **"EDGAR"**: not mentioned anywhere in the paper.

## 5. Other directly relevant sources

I did not find another primary source specifically about LLM-evolved
acquisition functions or BO heuristics that seeds with a deliberately
diverse hand-written baseline set (spanning UCB/EI/EHVI/novelty-style
families) the way this repo does. The closest adjacent works surfaced by
search (EvolCAF-style cost-aware AF evolution, LLaMEA-BO, "LLMs for Bayesian
Optimization in Scientific Domains") were not fetched in full given time
constraints and are flagged here only as leads, not verified claims — I'm
deliberately not padding this report with secondhand characterizations of
papers I didn't actually read.

## Recommendation (revised after reading the Harris/Tilbury paper directly)

**Four real primary sources now checked, all consistent on one point: none
of them seed with a large (5+), strategy-diverse hand-written set the way
this repo's 7 seeds do. FunBO/FunSearch/AlphaEvolve use exactly one seed;
the Harris-group system uses exactly two, and even those two aren't
strategically diverse — they're the same functional family (Gaussian bump)
differing only in peak count.** This is a more consistent pattern across
independent sources than I had before reading the fourth paper, and it
should shift the recommendation, not just add a footnote.

What's now established across all four sources:
- **Seed count is small everywhere in the literature** (1–2), never the 7
  this repo uses.
- **Seed diversity, where it exists at all (Harris group's 2), is minimal
  and domain-specific** (single- vs. double-peaked, a distinction that
  pre-exists in the tuning-curve literature) — not a deliberately
  constructed "cover the strategy space" set the way trust_only /
  novelty_only / fixed_ucb / ucb_plus_novelty / phase_decaying_ucb /
  ehvi_approx / mc_hvi_approx are.
- **All four systems get diversity from evolution + islands, not from the
  seed set.** FunBO/FunSearch/AlphaEvolve replicate one seed across
  islands; the Harris-group system replicates two seeds across 8 islands.
  Diversity in the *population* emerges from independent evolutionary
  trajectories and LLM stochasticity acting on that small seed, not from
  the initial programs themselves spanning many strategies.
- **No source anywhere claims or ablates that more/diverse hand-written
  seeds converge faster.** The Harris-group paper's own ablations (Fig.
  2d-f) test graphical diagnostics, gradient-descent parameter estimation,
  and the parameter-estimator lever — never seed count or seed diversity.

**Given this, I'd now revise my earlier "keep all 7, no source argues
against it" toward something more pointed: the literature's consistent
practice is a small, minimal seed set, with the real diversity-generating
work done by evolution + islands (which this repo doesn't currently have —
`run_evolution` is a single population, not an island model). That's worth
you weighing directly: either (a) keep 7 seeds but recognize this repo is
doing more upfront hand-engineering of the starting point than any of these
four systems did, compensating for not having an island architecture, or
(b) consider whether an island-style architecture (this repo doesn't have
one — `run_evolution` runs one flat population) would let you shrink back
to 1-2 seeds and rely on independent-trajectory diversity instead, closer to
what every one of these four sources actually does.**

I'm not making that call for you — it's a real architectural fork (seed-set
size trades off against whether you adopt islands), and the answer depends
on effort you're willing to invest in an island model vs. hand-authoring
more seeds, which is a decision only you can make. But I want to flag
plainly: my earlier "prior sources don't give a clean answer" was written
before I'd read the fourth (Harris-group) source directly, and having read
it, the pattern across all four is actually fairly clean — small seed sets,
diversity from islands/evolution, not from the seed set. This repo departs
from that pattern more than I'd initially characterized.

Practical, narrower recommendations that don't require an architecture
change:
- **The 3 seeds missing `term_weights`** (`phase_decaying_ucb`,
  `ehvi_approx`, `mc_hvi_approx`) are a bigger problem in light of this
  literature than I'd flagged before: in these four systems, a seed's whole
  point is to be inherited from during evolution (crossover/mutation over
  its actual content). This repo's mock-mode crossover silently discards
  exactly those 3 seeds' logic and substitutes a random program whenever
  they're drawn as a parent (`parent_a.get("term_weights") or
  random_program(rng)` in `evolve_af.py`'s `make_child`) — meaning 3 of the
  7 seeds can only ever survive as static elites in mock mode, never
  actually get bred from, which is a much sharper contradiction of "why
  have this seed" once you see how load-bearing seed inheritance is in
  every source reviewed here.
- If you keep all 7, backfilling `SEED_TERM_WEIGHTS` for the missing 3 (or
  accepting they're real-LLM-mode-only seeds, since real-LLM crossover reads
  `code` directly rather than `term_weights` and doesn't have this gap)
  would bring this repo closer to "seeds are actually inherited from," which
  every one of the four sources treats as the entire point of seeding.
