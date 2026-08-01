# LLM-Evolved Acquisition Functions (`llm_af_evo/`) — Full History Log

This document is a comprehensive, from-scratch explanation of everything done in the
`llm_af_evo/` project, reconstructed from the code, results files, and findings
documents in this directory (no chat history exists for this work — it predates
the currently-tracked conversation). File timestamps span **July 20 – July 31**.
The directory is **not** a git repo, so there is no commit history; the chronology
below is inferred from file modification times and content.

**This document covers the pre-v2 era (through ~Jul 27): fitness-generation history
(2a → 2b → margin), the eight mechanism gates in `L_COATINGS_FINDINGS.md`, and the
compose_batch/DA-COREG dead ends.** For the detailed, per-run numeric breakdown of
the later v2 evolution runs (Jul 28–31: Runs 1–4, the hint-bug postmortems, the mAb
noise diagnostic, and the 3-seed re-validation), see the companion document
[llm_evolved_afs_comprehensive_log.md](llm_evolved_afs_comprehensive_log.md) in this
same directory — that document was independently verified line-by-line against the
`evolution_runs/*/final_population.json` and `af_code_logs/*.py` files on 2026-08-01
and found to be accurate.

Everything is written assuming you're picking this up cold — including background
on *why* each step happened, not just what the numbers were. It also documents
failures and dead ends explicitly, since those carry as much information as the
wins.

---

## 1. What is this project actually trying to do?

In the sibling project (`ls_na_egbo/`, the LS-NA-EGBO benchmark), the acquisition
strategy used by the strongest baseline (`mo_egbo_novelty`) is a hand-designed
pipeline: **qLogNoisyExpectedHypervolumeImprovement (qLogNEHVI)** scores candidates,
**U-NSGA-III** generates/optimizes the candidate pool, and a **novelty-weighted
selection** rule picks the final batch. `llm_af_evo` asks a different question:
*can an LLM, acting as an automated program-search agent, discover an acquisition
function that beats this hand-designed pipeline — by writing and evolving actual
Python scoring code, not by prompting for formulation ideas?*

This is explicitly modeled on a small family of published "LLM-as-evolutionary-
mutator" systems, cited directly in the code and research docs:

- **FunBO** (Aglietti et al., arXiv:2406.04824) — LLM-evolved acquisition functions
  for standard (single-objective) BO. This is the closest precedent; `llm_af_evo`
  follows its fitness-evaluation philosophy (see §3) directly.
- **FunSearch** (DeepMind) — LLM evolves scoring functions for combinatorial math
  problems, program-database-style search.
- **AlphaEvolve** (DeepMind) — general LLM-driven code evolution.
- An unnamed UCL/Harris-lab system for equation discovery in neuroscience, cited
  specifically for its choice of **seed population size** (see §5).

### The unit being evolved

The evolved artifact is a Python function with signature:

```python
def score_pool(context: dict) -> list[float]
```

It receives one candidate pool per campaign batch and returns a score per
candidate; scores are always converted into a batch by **fixed top-k selection**
(`select_batch`) — batch *selection* is never itself evolved, only the *scoring*
function is. (This constraint was deliberately tested and held — see the
"selection gate," Part 3 below.)

`context` (defined in `af_interface.py` / `af_interface_v2.py`) is a
self-documenting dict so the LLM doesn't need external docs to write valid code:

- `pool`: candidate list, each with `x` (the raw formulation) and `gp_posterior`
  (per-objective GP mean/std, **pre-flipped so higher is always better** — this
  matters because it removes an entire class of "did I get the min/max direction
  right" bugs from LLM-generated code)
- `X_obs` / `Y_obs`: campaign history so far
- `objective_names`, `pareto_front`, `ref_point`: standard MOO bookkeeping
- `campaign`: progress state — current step, budget, campaign progress fraction,
  and `stagnant_batches` (how many recent batches made no Pareto-front progress) —
  included so the LLM can write *adaptive* strategies (e.g. explore harder when
  stagnant) without needing to track state itself.

### The evolutionary loop

Standard **mu+lambda elitist evolution**: a population of candidate `score_pool`
implementations, tournament selection of parents, then either a **mock mutator**
(deterministic, weak, LLM-free — for offline/no-network testing) or a **real LLM
mutator** (Ollama, e.g. `qwen3-coder:30b`) that rewrites/crosses-over the code.
Fitness = campaign performance metric minus `gamma * LOC` (lines-of-code) as a
complexity penalty, to discourage the LLM from just writing long, overfit special-
case code.

`mock_mutator.py` is worth flagging explicitly: **it is not a stand-in for the real
LLM mutator's logic** — it's a deliberately *weak* term-weight-dict crossover/
mutation, intentionally designed to not be able to beat the baseline on its own.
The purpose is methodological: if a mock-mutator-evolved AF ever did beat the
baseline, that would suggest the "win" came from lucky random search rather than
genuine LLM code understanding — so the mock mutator exists as a sanity floor,
not a cost-saving substitute.

### Two engine generations

| | v1 | v2 |
|---|---|---|
| Interface | `af_interface.py` | `af_interface_v2.py` |
| Evolution engine | `evolve_af.py`, `evolve_af_2b.py` | `evolve_af_v2.py` (63KB, most developed) |
| Seed population | 7 hand-written, strategically diverse seeds (trust_only, novelty_only, ehvi_approx, phase_decaying_ucb, etc.) | 1 hand-written seed (`trust_only`) + 8 one-line strategy hints the LLM implements itself at generation 0 |

The move from v1 to v2 was a direct, literature-motivated correction (see §5) —
not a bug fix, a philosophy change about how much of the design space to hand the
LLM up front versus let it discover.

---

## 2. Chronological narrative

**Jul 20–21 — Bootstrap.** Training-set generation (`generate_training_set.py`),
a first mock-only smoke test (`run_mock_subset.py`), an oracle-HV sanity check
(`diagnostic_true_oracle_hv.py`), and a sweep to calibrate the complexity-penalty
coefficient `gamma` (`sweep_lambda.py` → `sweep_lambda_summary.json`, settling on
`gamma* = 10000`... more precisely, this sweep calibrated the penalty weight used
in early fitness scoring, win_rate 0.678 at that setting).

**Jul 22–24 — First real evolution run, and an early failure mode.** `evolve_af.py`
(v1, first-generation fitness — see §3) ran a full evolution. It hit a **rank-
equivalence collapse**: zero fitness movement across 20 generations / 160 LLM
calls, because many structurally different AFs were producing identical relative
rankings and thus identical fitness under the metric used at the time. This was
diagnosed and fixed via **selection-signature deduplication** (`validate_2b_seed.py`,
`validate_l1.py`, `run_2b_diagnostic.py`) — tracking not just the fitness score but
a hash of which candidates were actually selected, so genuinely-different-behavior
AFs could be distinguished even when their scores tied.

**Jul 25–27 (early) — The mAb pilot gauntlet.** A rapid sequence of pilot
experiments testing individual mechanisms in isolation on the mAb domain:
composition, generation, batch-size, UNSGA3-pool, MC-integration, and baseline-
decomposition pilots (full list and results in §4). Every one of these closed
**negative** on mAb — no individual mechanism swap beat the qLogNEHVI+UNSGA3+
novelty baseline. `ada_coatings_oracle.py` also appears here, beginning the pivot
to a second domain (coatings) to check whether mAb's null results were domain-
specific or general.

**Jul 27 (late)–30 — Coatings domain, batch-composition family, synthetic
positive controls.** A parallel line of work explored *jointly* composing a whole
batch at once (`compose_interface.py`, `compose_strategies.py`, `compose_sandbox.py`,
`da_coreg.py` — a coregionalized multi-task GP surrogate) rather than scoring
candidates independently. This was tested first on **synthetic positive-control**
problems (ZDT1/DTLZ2 via `synthetic_mo_oracle.py`) specifically so that a failure
could be caught cheaply before spending real-domain compute — and it was caught:
compose_batch and DA-COREG both **lost** on their synthetic gates and were never
carried to the real domains. Two literature-survey documents were written in this
window: `research_fitness_alternatives.md` (surveying whether cheaper alternatives
to full-campaign-replay fitness exist — conclusion: no) and
`research_seed_population.md` (surveying how FunBO/FunSearch/AlphaEvolve/the
Harris-lab system size their seed populations — conclusion: all of them use 1–2
seeds, motivating the v1→v2 redesign). `L_COATINGS_FINDINGS.md` (35KB) was compiled
Jul 27, synthesizing all findings up to that point into Parts 1–8 (§3 below).

**Jul 28–30 — v2 rebuild.** `evolve_af_v2.py` / `af_interface_v2.py` built around
the minimal-seed-population redesign. Larger, 100-campaign training sets generated
for both domains (`training_logs_coatings_100`, `training_logs_mAb_100`), and real
(non-mock) LLM evolution runs launched and stored under `evolution_runs/`.

**Jul 31 — Chasing an anomaly, last day of work.** The v2 coatings run produced a
puzzling result: six structurally different AFs, all using nonzero uncertainty
weighting, landed on the **exact same win-rate** (52/75) despite having different
selection-signature hashes (i.e., they actually behaved differently campaign-to-
campaign, but tied in aggregate score). Four diagnostic scripts were written same-
day to chase this down (`diagnose_mu_sigma_dominance.py`, `diagnose_win_overlap.py`,
`mab_noise_diagnostic.py`, `validate_population_2b.py`) — see §6. This is the last
recorded activity in the directory (files up to 22:09).

---

## 3. The fitness function — three generations, and why each was replaced

Getting the *training signal* right turned out to be its own multi-week problem,
independent of the acquisition-function search itself.

**Generation "2a"** (`fitness_common.py`, used by `evolve_af.py`): a cheap,
single-step counterfactual-regret proxy — `HV_pred_improvement + λ * sigma_norm`
against a GP-only exploration-credit estimate, with `λ` calibrated by
`sweep_lambda.py`. This was **found invalid** on three independent grounds:
- **Gameable**: degenerate pure-uncertainty AFs (i.e., AFs that just chase GP
  variance and ignore everything else) could score near-optimally under this
  metric without doing anything a real campaign would reward.
- **Off-policy**: the Pareto front used to score a candidate AF reflected the
  *baseline's* trajectory, not the trajectory the candidate AF would actually
  produce if it were driving the campaign.
- **Empirically uncorrelated with the thing it's supposed to proxy**: measured
  Spearman correlation with true full-campaign HV was ρ = 0.125 (p = 0.645) —
  statistically indistinguishable from zero.

**Generation "2b"** (`evolve_af_2b.py`, `full_replay.py`): the fix — full-campaign
replay. Each candidate `score_pool` is actually installed as the acquisition
function and run through a complete campaign against real training-log data, and
its final HV is compared to the baseline's HV *on the same campaigns*. This is
more expensive but on-policy and non-gameable. A literature check
(`research_fitness_alternatives.md`) confirmed this matches what FunBO itself does,
and that cheaper alternatives (off-policy evaluation / bandit-style estimators)
don't transfer here because there's no well-defined action-propensity model and
campaign trajectories are strongly non-i.i.d.

**v2's margin-based fitness** (`evolve_af_v2.py`): 2b's win-rate (fraction of
campaigns won, a binary per-campaign count) had a resolution problem — with only
`n_campaigns + 1` distinguishable values, the LOC complexity penalty could
sometimes *invert* the true ranking between two AFs whose win-rates tied but whose
actual margins differed. v2 replaced win-rate with **mean relative HV margin** as
the primary fitness signal, preserving the on-policy full-replay approach from 2b
but with continuous resolution.

---

## 4. `L_COATINGS_FINDINGS.md` — the central results document (Parts 1–8)

This is the main synthesis document (compiled Jul 27) and reads as **eight
independent "gates,"** each testing one candidate mechanism, almost all of which
closed negative.

**Part 1 — mAb domain, full evolution run.** After fixing 2a→2b fitness and the
rank-collapse bug, a 20-generation / 160-LLM-call run produced 40 distinct AFs,
verified (by inspection) to actually use the intended vocabulary (Pareto front,
reference point, GP posterior mean). On 20-campaign held-out validation, **none
beat baseline**: best result was a `trust_only`-family AF at +2.4% (p = 0.50,
not significant); `ehvi_approx` reached +1.0% (p = 0.70).

**Part 2 — Coatings domain, replicated.** Uses the MacLeod et al. 2022 real self-
driving-lab dataset (253 real samples, 4 real campaigns), snapped to a discrete
pool specifically to avoid a documented 193× GP-extrapolation failure mode seen on
a related dataset. Getting this domain right required fixing three separate bugs
first:
1. A fake 3-objective framing — two of the three "objectives" were columns with
   r = 1.0 correlation, meaning it was really a disguised single-objective problem.
   Corrected to a genuine 2-objective framing (conductivity vs. conductance std,
   r = 0.0032).
2. Hardcoded `["max", "max", "min"]` objective directions that would have silently
   corrupted results on an all-max oracle if reused elsewhere.
3. CPU/BLAS and pymoo-UNSGA3 non-determinism — the *same* nominal run produced
   p = 0.033 on one execution and p = 0.82 on another. Fixed via explicit BLAS
   environment variables set pre-import plus explicit UNSGA3 seeding (a plain
   `torch.manual_seed` alone was insufficient).

After these fixes, three independent replicates gave: `trust_only` +4.8%
(significant in 2 of 3 replicates), `phase_decaying_ucb` +4.6%, `evolved_adaptive`
+4.4%, `ehvi_approx` +0.0% (never significant in either direction on this domain).

**Combined domain picture**: front/HVI-style AFs tie the baseline on *both*
domains. Exploitation-flavored AFs tie on mAb but show a modest, replicated edge
(~+4.5%) on coatings. Adding adaptivity on top of static exploitation adds nothing
further in either domain. Note: an earlier draft had claimed this domain
difference was driven by landscape sparsity/dimensionality — that specific causal
claim was **explicitly checked and retracted** when the two supporting numbers
didn't hold up under direct re-measurement.

**Parts 3–8 — mechanism gates** (mostly mAb + DTLZ2/ZDT1 synthetic + coatings):

| Part | Mechanism tested | Verdict |
|---|---|---|
| 3 | Novelty-weighted vs. fixed top-k batch selection | Closed negative — no gap movement either way |
| 4 | Geometry-only candidate generation (replacing qLogNEHVI + UNSGA3 with e.g. LHS-unexplored) | Closed negative — actively lost, −3.7% |
| 5 | Batch-size ablation, q ∈ {1,2,3,5,10} | Closed negative/refutation — no monotonic trend, rules out "redundancy discounting" as the mechanism behind any observed edge |
| 6 | Monte-Carlo integration over posterior uncertainty (`mc_hvi_approx`) | Closed negative |
| 7 | Genuinely joint batch composition (`compose_batch`) + DA-COREG, 2×2 on ZDT1 | Closed negative — compose_batch **actively lost** (p = 0.0005–0.0094); an uncertainty-aware variant didn't recover it; never carried to real domains |
| 8 | DA-COREG alone, isolated on DTLZ2 (positive-control sanity check) | **Lost badly** (p ≈ 1.9×10⁻⁶); three bug hypotheses ruled out — not a silent fallback, not a column-ordering bug, and DA-COREG's actual GP fit was *better* than the independent-GP baseline. Leading remaining hypothesis: correlated joint MC sampling inside qLogNEHVI compresses acquisition-value discrimination when paired with a MultiTaskGP surrogate — plausible but not proven; deprioritized and not pursued further |

**Overall verdict stated in the document**: *"qLogNEHVI's joint, MC-integrated,
gradient-optimized treatment of the whole batch is doing work that no
decomposition or component substitution tested here... recovers."* The one open
thread left unresolved: why coatings shows a real exploitation edge and mAb shows
none — with only two domains tested, candidate confounds (dimensionality, pool
sparsity, real vs. synthetic noise, continuous vs. categorical variables) can't be
disentangled.

---

## 5. Every pilot / ablation experiment, with numbers

| Experiment | File | Result |
|---|---|---|
| Composition gate, mAb | `composition_pilot_results.json` | trust_only_topk +1.7%, p = 0.39 (null) |
| Composition gate, coatings | `composition_pilot_coatings_results.json` | trust_only_topk +7.5%, p = 0.002 (significant) |
| Generation gate | `generation_pilot_results.json` | gen_lhs −2.4%, p = 0.23 (trending negative) |
| Batch-size ablation | `batch_size_ablation_results.json` | bs=1: +0.2%, p = 0.92 → bs=2: +5.2%, p = 0.16; no monotonic trend across q |
| Coatings replicates | `coatings_replicates_results.json` | trust_only +7.5%, p = 0.002 (this replicate) |
| Coatings generalization | `coatings_generalization_results.json` | 20 campaigns, budget 40, seed 42 (config/generalization check) |
| Synthetic compose, ZDT1 | `synthetic_compose_zdt1_results.json` | baseline_da_coreg +0.07%, p = 0.81 (null); compose_indep −0.99% (loss) |
| UNSGA3 pool pilot | `unsga3_pool_pilot_results.json` | trust_only_unsga3 −3.4%, p = 0.11 |
| MC-HVI pilot | `mc_hvi_pilot_results.json` | mc_hvi_approx_topk −0.17% (null) |
| Baseline decomposition, coatings | `baseline_decomposition_coatings_results.json` | mo_egbo_real (novelty weighting removed) +5.4%, p = 0.006 — shows novelty selection alone isn't the source of the baseline's edge |
| Baseline decomposition, mAb | `baseline_decomposition_mab_results.json` | qnehvi_plain (no UNSGA3, no novelty) −9.1% — loses badly without the full pipeline |
| DA-COREG, DTLZ2 | `da_coreg_dtlz2_results.json` | −3.8%, 0/20 wins, p = 1.9×10⁻⁶ |

---

## 6. Failed / abandoned approaches (explicit list)

- **2a fitness** (GP-only single-step proxy) — abandoned as gameable and
  uncorrelated with true campaign outcome; replaced by full-campaign replay (2b).
- **`compose_batch` joint-composition family** (`compose_interface.py`,
  `compose_strategies.py`, `compose_sandbox.py`, `da_coreg.py`) — actively hurt
  performance on its synthetic positive-control gate; correctly never carried to
  a real domain (the synthetic-gate-first methodology worked exactly as intended
  here — it caught the failure before real-domain compute was spent).
- **DA-COREG** (coregionalized multi-task GP surrogate) — lost significantly on
  the DTLZ2 positive control despite fitting the data *better* (lower held-out
  NLL) than independent per-objective GPs; not pursued further.
- **Geometry-only candidate generation** (`gen_lhs_unexplored`,
  `gen_perturb_front_extremes`, `gen_half_exploit_half_explore` in
  `gen_interface.py` / `gen_sandbox.py`) — failed to replace gradient-based
  `optimize_acqf`; the pure-coverage variant actively hurt performance the most.
- **7-seed hand-diverse population design** (v1's `af_interface.py`) — not wrong
  exactly, but abandoned once the seed-population literature review
  (`research_seed_population.md`) found every cited precedent system uses 1–2
  seeds, not 7; replaced by v2's minimal-seed design.
- **Original 6 UCB-style strategy hints in `af_interface_v2`** — on the real
  mAb-100 run, all 8 surviving AFs collapsed into simple UCB variants; 3 of the 6
  original hints (novelty_only, ehvi_approx, mc_hvi_approx) didn't survive
  selection and were replaced with hints encoding genuinely new mechanisms (DPP
  diversity, greedy local penalization, Pareto-membership trust regions).
- **3-objective coatings framing** — discovered to be a near-degenerate disguised
  single-objective problem (r = 1.0 between two of three columns); discarded.
- **`torch.manual_seed` alone for reproducibility** — insufficient on its own;
  needed BLAS environment variables set pre-import plus explicit pymoo UNSGA3
  seeding to get stable p-values across repeated runs.

---

## 7. July 31 — the last thing worked on (unresolved)

The final day's work was diagnostic, not evolutionary — chasing an anomaly in the
v2 coatings run (`evolution_runs/run_v2_coatings_gamma001_v2`): **six structurally
different AFs, all using nonzero uncertainty weighting, tied at the exact same
win-rate (52/75)**, despite having different `selection_signature` hashes (proof
that they actually behaved differently campaign-to-campaign — this wasn't the
same rank-collapse bug from Jul 22–24, since the signatures differed).

Four scripts were written same-day to pin this down:

- **`diagnose_mu_sigma_dominance.py`** — tests whether GP posterior sigma
  (uncertainty) is near-constant across the candidate pool, which would make any
  uncertainty-weighted AF degenerate into `trust_only`'s ranking regardless of its
  actual weight — versus a threshold effect where mu (mean) only dominates at the
  particular weight values the hint-derived AFs happened to land on.
- **`diagnose_win_overlap.py`** — re-runs the saved candidate code against all 75
  training campaigns and computes Jaccard overlap of *which* campaigns each AF won,
  to distinguish "these AFs are winning the exact same campaigns" (campaign-
  difficulty dominates, i.e. some campaigns are just easy regardless of AF) from
  "different campaigns, same count by coincidence" (a metric-resolution collision).
- **`mab_noise_diagnostic.py`** — a related but separate investigation into the
  mAb domain's noise floor: the standard error of the mean margin (~0.057) was
  larger than the observed gap between top AFs (~0.011), raising the question of
  whether this was pipeline noise (GP-fit seed, UNSGA3 seed) or genuine domain
  noise. GP-fit and UNSGA3 seeds were decoupled and varied independently; results
  (`mab_noise_diagnostic_results.json`) show paired margins ranging from −0.31 to
  +0.79 across 15 seeds **on a single fixed campaign** — confirming substantial
  pipeline-level seed noise exists and does not cancel out even under shared
  seeding.
- **`validate_population_2b.py`** — updated same day, presumably to re-validate
  the population against the newly-understood noise characteristics.

The `mab_noise_diagnostic` finding directly motivated new code in
`evolve_af_v2.py`: `N_FITNESS_SEEDS_DEFAULTS` (averaging fitness over multiple
seeds, default 3) and `FITNESS_STAT_DEFAULTS` (using median instead of mean for
mAb specifically) plus a gamma recalibration — partial mitigations, not a full
fix. The last file touched in the whole directory is a run using this fixed
config: `evolution_runs/run_v2_mAb_gamma001_fixed` at 22:09 Jul 31.

**Important gap**: I could not find a written conclusion for the
mu/sigma-dominance or win-overlap investigations specifically — no dated document
newer than `L_COATINGS_FINDINGS.md` (Jul 27) synthesizes what they found. The
`evolve_af_v2.py` code changes suggest the noise-related findings were acted on,
but there's no equivalent "Part 9" writeup, and the last run
(`run_v2_mAb_gamma001_fixed`) has no accompanying results analysis anywhere in the
directory. **This work appears to have stopped mid-investigation, not
concluded.**

---

## 8. Open questions (as of the last file touched)

1. **The central unresolved question from `L_COATINGS_FINDINGS.md` itself**: why
   does coatings show a real, replicated ~+4.5% exploitation-AF edge while mAb
   shows none at all? With only two domains tested, none of the plausible
   confounds (dimensionality, candidate-pool sparsity, real vs. synthetic noise,
   continuous vs. categorical variables) can be isolated.
2. The Part 8 DA-COREG loss mechanism (correlated joint MC sampling inside
   qLogNEHVI compressing acquisition discrimination when paired with a
   MultiTaskGP) is a plausible but **unproven** hypothesis — deliberately
   deprioritized rather than run to ground.
3. Whether the Jul 31 mu/sigma-dominance and win-overlap diagnostics reached any
   conclusion is unclear — no write-up exists for them specifically.
4. mAb's noise floor (mixed pipeline + domain noise, pipeline noise confirmed
   substantial via `mab_noise_diagnostic.py`) is only partially mitigated by the
   `N_FITNESS_SEEDS_DEFAULTS=3` averaging added afterward — increasing
   `n_campaigns` is flagged as the remaining unaddressed lever, not yet tried.
5. `run_v2_mAb_gamma001_fixed` (the last `evolution_runs/` entry, 22:09 Jul 31)
   has no accompanying analysis anywhere in the directory — its outcome is
   currently undocumented.

---

*Compiled retrospectively from directory contents (no git/chat history available
for this project). If you resume this work, the two highest-leverage next steps
implied by the open questions above are: (1) write up whatever
`run_v2_mAb_gamma001_fixed` actually produced, since it's the last run and
currently has zero analysis attached to it, and (2) decide whether the
mu/sigma-dominance investigation needs to be finished or can be closed as "noise,
not a real finding" — right now it's neither.*
