# Research note: is there something better than full-campaign replay (2b) for training-signal quality?

## Scope and method

Primary sources fetched and read directly (arXiv HTML/abstract pages, PDF where
HTML unavailable), not blog summaries, except where explicitly flagged as
unread. Question: given that fitness signal "2a" (single-decision-point
counterfactual scoring against a baseline's logged trajectory) was judged
invalid for being off-policy — the front/history state at each logged
decision point reflects the *baseline's* accumulated trajectory, not what the
candidate AF's own trajectory would look like from the start — and that
`full_replay.py`'s current fitness ("2b") fixes this by re-simulating entire
campaigns (real GP fit + qLogNEHVI/optimize_acqf candidate generation kept
identical to the baseline, only the final per-candidate scoring/selection
step swapped for the candidate AF), is there a technique in the literature
that preserves on-policy validity at lower cost than full re-simulation?
Four bodies of work were checked: (1) FunBO itself, the closest prior
programs-database-evolution system for AFs; (2) off-policy evaluation (OPE)
in contextual bandits/RL; (3) BO-specific meta-learning of acquisition
functions via RL; (4) truncated/n-step rollout techniques as a general
RL/OPE concept. Each section below reports only what was actually verified
in the source; where a source could not be fetched in full, that is stated
plainly rather than filled in from secondhand characterization.

## 1. FunBO — Aglietti et al., arXiv:2406.04824 (fetched via arxiv.org/html
HTML render, Section 3 and Limitations)

This is the directly relevant precedent: `evolve_af.py`'s own docstring
names FunBO explicitly, and it is the one prior system in this space that
evolves acquisition functions via LLM-driven programs-database search, the
same structural pattern this repo uses.

- **Evaluation mechanism, verbatim from the paper (Section 3)**: each
  LLM-generated candidate `h_τ` is "(i) checked to verify it is correct,
  i.e., it compiles and returns a numerical output; (ii) scored based on the
  average performance of a BO algorithm using `h_τ` as an AF on `𝒢_Tr`" — a
  training set of objective functions. This is a **full simulated BO loop
  per candidate, per training function**, not a cheaper single-step or
  partial proxy.
- **Scoring formula (their Eq. 1)**: score is an average over `j ∈ 𝒢_Tr` of
  a term combining (a) how close the BO run using `h_τ` got to the true
  optimum relative to the run's starting point, and (b) how many trials it
  took to converge (a "budget-efficiency" term). Both components require
  running the BO loop to completion (or to a fixed trial budget `T`) on
  each function in the training set — there is no single-step or
  truncated-episode variant in the formula itself.
- **Episode length**: `T = 30` trials per BO loop in most reported
  experiments (Appendix C / Figure 9's `num_trials`).
- **Objective functions**: `𝒢_Tr`, a **training set of multiple objective
  functions** (the paper's experiments use held-out synthetic/benchmark
  test functions, not one real oracle) — every candidate is scored on the
  *average* over that whole set, meaning FunBO's per-generation cost is
  (population size) × (training-set size) × (T-step BO loop), a heavier
  per-candidate cost structure than this repo's 2b (which replays a fixed
  set of logged campaigns once per candidate, not re-simulating against a
  *training set of synthetic objectives* the way FunBO does).
- **Cost is explicitly acknowledged as a limitation, with no cheaper
  alternative proposed.** Direct quote from the paper's own Limitations
  section: "a potential limitation of FunBO is the computational overhead
  associated with running a full BO loop for each function in `𝒢`, which
  significantly increases the evaluation time of every sampled AF," and
  that this "limits the scalability of FunBO for larger sets `𝒢` and
  hinders its application to more complex optimization problems." **The
  paper does not propose or discuss any cheaper single-step, truncated, or
  proxy-based alternative to full BO-loop evaluation** — it names the cost
  problem and stops there.
- **Bottom line for this research question**: FunBO, the closest published
  analogue to this repo's evolution loop, uses exactly the same category of
  fitness signal this repo's 2b now uses (full simulated on-policy rollout
  per candidate), regards it as expensive, and offers no alternative. This
  is evidence *against* the existence of a published cheaper-but-equally-valid
  proxy in this specific line of work, not evidence for one.

## 2. Off-policy evaluation (OPE) in contextual bandits — Dudík, Langford,
Li, "Doubly Robust Policy Evaluation and Learning," arXiv:1103.4601
(fetched via arxiv.org/pdf render)

- **Core requirement: known (or estimable) logging propensities.** The
  doubly-robust (DR) estimator combines a reward model with inverse
  propensity weighting, and its inverse-propensity term requires "knowledge
  of the logging policy's propensities" — i.e., the probability that the
  policy which generated the logged data would have taken the specific
  logged action, for every logged decision.
- **Action space: discrete/finite.** The paper's own experimental setting
  (Section 5.1) is multiclass classification with bandit feedback — a
  finite, enumerable action set (one action per class). Nothing in the
  fetched content discusses continuous or combinatorial (batch/slate)
  action spaces.
- **Validity requirements stated in the paper**: an overlap/support
  condition — "the logging policy must assign positive probability to all
  actions" the evaluation policy might select, so the inverse-probability
  weights stay bounded — plus (implicitly, Section 2) i.i.d. contexts. The
  paper's own bias analysis (Section 3) states that violations of the
  support assumption produce bias, and its variance analysis (Section 4)
  shows that propensities close to zero blow up variance.
- **Why this does not transfer to this repo's setting, concretely**:
  1. **Action space mismatch.** This repo's "action" at each decision point
     is a *batch selection over a continuous, GP-posterior-scored candidate
     pool* generated by qLogNEHVI/optimize_acqf + UNSGA3 — not a draw from a
     small finite action set. DR/IPS need either a known finite action
     distribution or an estimable density for continuous actions; neither
     this repo's baseline (`strategy_mo_egbo_novelty`) nor any real BO
     acquisition strategy exposes an explicit, normalized probability
     distribution over the candidate pool — it deterministically (or
     near-deterministically, modulo stagnant-batch conditioning) picks a
     top-k or novelty-weighted batch. There is no "logging propensity" to
     plug into the IPS/DR weight at all, let alone one that's bounded away
     from zero for whatever alternative batch the candidate AF would pick.
  2. **Non-i.i.d., path-dependent contexts.** DR/IPS assume each logged
     (context, action, reward) tuple is drawn from a stationary,
     context-conditionally-independent process. In a BO campaign, the
     "context" at decision point *b* (the GP posterior, the current front)
     is a deterministic function of every prior decision back to the seed
     — exactly the sequential, trajectory-dependent structure that made 2a
     invalid in the first place. Bandit-style OPE is built for single-shot
     or context-per-round problems where the context does not depend on the
     evaluated policy's own prior choices; it is not designed to correct
     for an entire trajectory's worth of accumulated distributional shift,
     which is precisely what a multi-batch BO campaign produces.
  3. **Batch/slate structure compounds this.** Even bandit OPE variants
     built for slates (which this fetched paper does not cover) typically
     assume a known slate-generation policy with tractable propensities;
     BO's batch is jointly selected by GP-posterior-driven optimization,
     not sampled from any explicit stochastic policy at all.
  - **Conclusion**: DR/IPS-style OPE, as characterized by this paper, does
    not transfer cleanly to this repo's problem. The mismatch is not a
    minor implementation detail — it is a structural mismatch on two of the
    method's foundational assumptions (known/estimable propensities over a
    well-defined action distribution; i.i.d. or per-round-independent
    contexts), both of which fail for sequential batch BO with a
    deterministic GP-driven selection rule.
- **Precup, Sutton, Singh, "Eligibility Traces for Off-Policy Policy
  Evaluation" (ICML 2000) — NOT independently verified.** Multiple fetch
  attempts (arXiv abstract search — this paper predates arXiv and has no
  arXiv ID; a direct PDF mirror at incompleteideas.net; Semantic Scholar's
  paper page) all failed to return readable full text (certificate error,
  403, and an empty-content response respectively). I am not characterizing
  this paper's contents here because I could not verify them against the
  primary source in this session. What I can say without relying on it:
  the general point it is normally cited for — extending importance-sampling
  OPE to eligibility-trace/TD-style updates in sequential RL, which still
  requires known behavior-policy action probabilities at each step — would,
  if accurate, face the same propensity and non-i.i.d.-context problems
  identified above for the bandit case, likely worse, since the sequential
  RL formulation compounds importance weights multiplicatively along the
  trajectory (a well-known variance-blowup problem in that literature more
  broadly), but I am explicitly flagging this as unverified inference, not
  a sourced claim.

## 3. BO-specific meta-learning of acquisition functions via RL — Volpp et
al., "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian
Optimization" (MetaBO, arXiv:1904.02642, ICLR 2020; fetched via
ar5iv.labs.arxiv.org HTML render)

- **Reward signal**: negative simple regret per step, `r_t ≡ -R_t` where
  `R_t ≡ f(x*) - f(x_t+)` (Section 4, "Training Procedure"); a log-regret
  variant `r_t ≡ -log10(R_t)` is used when the true optimum is known
  (Appendix B.1).
- **Full episodes, run to completion, no partial-rollout alternative
  found.** Training rolls out "a batch of episodes in the inner loop" of
  policy-gradient (PPO) updates, with fixed episode length `T` equal to the
  optimization budget (Table 1: `T ∈ {30, 50}` depending on task). **Zero
  mention anywhere in the fetched text of truncated episodes, n-step
  returns, or bootstrapped value estimates as a substitute for full
  rollouts** — the one variance-reduction technique present is a learned
  value function `V^π(s_t)` used inside PPO's advantage estimation, but
  this still estimates return from the full *remaining* trajectory to `T`,
  not a bootstrapped return computed from a truncated horizon; it's PPO's
  standard variance-reduction baseline, not a cost-reduction device.
- **Objective functions during training are cheap proxies, not full
  simulation avoidance.** MetaBO trains on "a set of functions ℱ′ which
  capture relevant properties of ℱ but are much cheaper to evaluate,"
  concretely: randomly translated/scaled synthetic benchmarks
  (Branin/Goldstein-Price/Hartmann-3) for global optimization, physics
  simulations for a sim-to-real pendulum task, precomputed loss-surface data
  for HPO, and GP-prior-sampled functions for the "general function class"
  experiments. **This is a different cost-cutting axis than the one this
  repo can access**: MetaBO cuts cost by making the *objective evaluation
  within each step* cheap (synthetic function calls instead of a real wet-
  lab/expensive oracle), while still running the *entire sequential episode
  structure* in full. This repo's oracle (a fitted, logged experimental
  surface standing in for the real objective) plays the role of MetaBO's
  real-world `f` — but MetaBO's actual cost saving comes from training on
  a *different, cheaper proxy objective class* (`ℱ′` instead of `ℱ`), not
  from any shortcut to the episode-simulation structure itself. This repo
  already queries a cheap "oracle" (a fitted surrogate over logged data,
  not a live wet-lab run) per campaign step, so it has, in effect, already
  captured the axis of savings MetaBO exploits — it just also needs the GP
  fit + optimize_acqf machinery per step, which is not the axis MetaBO cuts.
- **Bottom line**: MetaBO is a second independent confirmation of the same
  pattern as FunBO — every RL/evolutionary approach to training a
  parameterized AF found in this search runs full simulated episodes to
  compute its training signal; none substitutes a cheaper partial-episode
  or single-step proxy for the *sequential* rollout itself. The cost-saving
  lever these systems actually use is a cheaper *objective function*
  (synthetic/GP-sampled), which is orthogonal to full-vs-partial rollout
  and not directly available to this repo (whose "real oracle" already
  plays that cheap-proxy role for whatever the true wet-lab objective
  ultimately is).

## 4. Truncated/n-step rollout as a general bias-variance technique

This is a well-established concept in RL generally (TD(n), TD(λ),
n-step returns, bootstrapping a value estimate for the tail of a
trajectory instead of simulating it to termination) — but I was not able to
verify, in the sources actually fetched during this session, a paper that
**applies this specifically to evaluating a batch-acquisition policy in
Bayesian optimization** or to programs-database/FunSearch-style AF
evolution. Neither FunBO nor MetaBO (the two directly relevant sources
fetched above) mention or use truncated/bootstrapped rollouts anywhere in
their evaluation loops — both were checked explicitly for this and both
came back with **no mention** of the concept. I did not find and did not
fetch a BO-specific paper proposing bootstrapped-tail rollouts as a cheaper
substitute for full-campaign simulation; I am flagging this as a gap in
what this session's search actually surfaced, not asserting the technique
doesn't exist anywhere in the wider literature.

What can be said about the general (non-BO-specific) technique, reasoning
from the standard bias-variance tradeoff it is known for rather than from
an unread source: bootstrapping the tail of a trajectory with a value
estimate (rather than simulating to the end) trades reduced compute for
introduced bias equal to the value estimate's own error, and this bias is
compounded, in this repo's specific case, by the same distributional-shift
problem that invalidated 2a: any bootstrapped "value" for the untaken
remainder of a campaign would have to come from *somewhere* — either (a)
the baseline's own logged continuation (which reintroduces 2a's exact
flaw for every batch after the truncation point, since the baseline's
front/history from batch k+1 onward reflects the baseline's own choices,
not the candidate AF's), or (b) a learned value function trained on
exactly the kind of full-replay data this repo is trying to avoid
generating, which is circular for a system whose whole cost problem is
generating that data in the first place. Both options reduce, on inspection,
to either re-introducing the off-policy flaw at the truncation boundary or
requiring the full-replay data anyway to train the bootstrap. This is
reasoning from the general RL concept, not a sourced claim about a specific
paper — flagged as such.

## Recommendation

**Full-campaign replay (`full_replay.py`'s 2b) is, on the evidence actually
gathered in this session, the closest thing to a validated state-of-the-art
approach for on-policy evaluation of a sequential batch-BO acquisition
policy without an expensive real-world oracle rollout — and the literature
searched here does not offer a cheaper-but-equally-valid alternative.**

Specifically:

1. **FunBO, the one directly comparable published system, uses the same
   category of signal (full simulated on-policy rollout per candidate) and
   explicitly names the cost as an unsolved limitation, without proposing
   or citing a cheaper valid alternative.** This is the strongest single
   data point: the closest prior art in exactly this problem area
   independently arrived at "run it in full, it's expensive, we don't have
   a fix for that."

2. **MetaBO (a second, independent AF-learning system) shows the same
   pattern**: full episode simulation, no partial/truncated/bootstrapped
   substitute anywhere in its training procedure. Its actual cost lever —
   training on cheaper synthetic/GP-sampled objective functions instead of
   the real objective — is a different axis than full-vs-partial rollout,
   and this repo already occupies that axis (its oracle is already a cheap
   stand-in, not a live expensive wet-lab query), so MetaBO's cost-cutting
   trick is not an additional lever available here.

3. **Contextual-bandit OPE (Dudík/Langford/Li's doubly-robust estimator) is
   the one concrete "smarter alternative" candidate this research question
   asked about directly, and it does not transfer, for two specific,
   sourced reasons**: (a) it requires known or estimable logging
   propensities over a well-defined action distribution, and BO's batch
   selection from a continuous GP-posterior-scored candidate pool has no
   such propensity — the baseline strategy is a deterministic (or
   near-deterministic) top-k/novelty-weighted rule, not a policy with an
   explicit, invertible action distribution; and (b) it assumes i.i.d. or
   per-round-independent contexts, whereas a BO campaign's context at step
   *b* is a deterministic function of every earlier decision — exactly the
   accumulated-trajectory dependency that made the single-step proxy (2a)
   invalid in the first place. Bandit OPE is built to correct for a
   different, weaker kind of distributional shift (a fixed context
   distribution, different action policy) than the one this repo's problem
   actually has (an entire trajectory of states that are *themselves*
   generated by the policy being evaluated). Applying it here would not
   fix 2a's flaw; it would just add unbounded/undefined importance weights
   on top of it.

4. **Truncated/n-step/bootstrapped rollout, the general RL technique this
   question asked about as a possible middle ground, was not found applied
   to this exact problem in any source fetched this session** (neither
   FunBO nor MetaBO use it). Reasoning from the technique's known general
   properties (not from a specific source): any bootstrap value for an
   un-simulated tail either reuses the baseline's own logged continuation
   (reintroducing 2a's exact flaw from the truncation point onward) or
   requires a learned value function trained on full-replay data anyway
   (circular, since generating that data is the cost problem 2b already
   has). Neither resolves to something both cheaper and validity-preserving
   given this repo's specific setup.

**Direct answer to the research question**: no, the literature surveyed
here does not support a technique that is both (a) cheaper than full
sequential re-simulation and (b) preserves on-policy validity for this
specific problem — a batch-acquisition policy operating on a continuous,
GP-posterior-scored candidate pool, whose state at every decision point is
a deterministic function of its own prior choices. Full-campaign replay
is expensive because the thing it is faithfully reproducing (real GP fit +
real candidate-generation optimization, on the policy's own resulting
trajectory) is itself expensive and load-bearing for validity — every
candidate alternative examined here either abandons that on-policy
property (2a, bandit OPE, baseline-bootstrapped truncation) or doesn't
actually save the cost this repo's replay incurs (MetaBO's cheap-objective
lever, already captured by this repo's oracle). If cost is the remaining
problem, the lever this session's sources point to is the one FunBO itself
identifies as unsolved — reducing evaluation cost without abandoning full
on-policy rollout is an open problem in this line of work, not something
with a published fix being missed here.
