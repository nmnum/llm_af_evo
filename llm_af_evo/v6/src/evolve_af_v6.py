"""
evolve_af_v5.py — copy of evolve_af_v4.py, extended with MULTI-DOMAIN
fitness (evaluate_af_multi_domain, see its own section header comment for
the full field-mapping rationale) to directly address the overfitting
problem v3/v4 both left open: v3's champion never generalized to mAb/
coatings, and a same-type synthetic-to-synthetic check (v3's champion vs
DTLZ2, a domain it never trained on) showed the SAME failure — the
champion was statistically indistinguishable from random scoring there.
That ruled out "synthetic-vs-real noise" as the explanation; the champions
were overfit to their exact training distribution, not just failing to
handle real-world observation noise.

WHY multi-domain fitness, not just more/better single-domain evaluation:
selecting only on one domain's fitness can't select FOR generalization —
a formula only has to fit that one domain's quirks to win. Training
against a combined score across tunable+coatings+mAb (v5's TRAINING_
DOMAIN_ORDER below), with DTLZ2 held out entirely for final validation
only (never trained on, mirroring the check that exposed the problem),
directly selects against exactly this failure mode instead of hoping a
champion generalizes after the fact.

See the af-evolution branch conversation (2026-09-01) for the full design
reasoning behind every choice: mean-0.5*std combination over worst-case/
min (a "loser's curse" argument — min-of-noisy-per-domain-estimates is
systematically pulled down by whichever domain got unlucky that
generation, not necessarily whichever the formula is genuinely worst on),
sequential gating tunable->coatings->mAb rather than parallel (cheapest/
cleanest domain first as a fast filter; NOT parallelized across domains —
this session's own CPU-contention bug, which silently broke ~half of a
real v4 run's LLM calls by competing for cores with background evaluation
work, is the concrete cautionary tale for why), and gamma starting at 0
(the old adaptive-gamma formula was calibrated against single-domain
margin scales and doesn't port to the new combined scale).

--- evolve_af_v4.py's own docstring, kept for context (mostly still true,
    "v4"/"af_interface_v4"/"evolve_af_v4.py" self-references below refer
    to the prior file, not this one) ---

evolve_af_v4.py — copy of evolve_af_v3.py (which is itself the tunable-
domain extension of evolve_af_v2.py), extended to expose EGBO's REAL
scoring signal to evolved AFs instead of only GP posterior mean/std.

WHY: run1/run2 (v3, tunable domain) showed the LLM repeatedly hand-
approximating hypervolume improvement from gp_posterior mean/std alone
(the "posterior_resampling"/"pareto_probability" MECHANISM_FAMILIES below)
under stagnation, and those approximations never beat the population's
real champion. Direct measurement (see full_replay.py's strategy_evolved_af
docstring / sandbox.py's pool_acq_value docs) showed why: EGBO's baseline
(strategy_mo_egbo_novelty) doesn't score candidates from mean/std at all —
it scores them with botorch's own qLogNEHVI acquisition value, a Monte-
Carlo integral over the FULL joint objective posterior, which is a
materially richer signal than anything decomposable into independent
per-objective mean/std features. That value is now exposed directly as
context["pool"][i]["acq_value_norm"] (min-max normalised over the pool,
same convention EGBO's own selection uses) — see af_interface_v4.py's
egbo_novelty_like seed and acq_value_progress_blend hint, both new in v4.
Validated directly before building any of this out: a trivial score_pool
returning acq_value_norm alone reached final_hv=10.92 on one heldout
tunable-domain campaign, essentially matching the real EGBO-novelty
baseline's range (10.86-10.94) — far closer than any hand-rolled
approximation reached across 128+ LLM calls in run1+run2 combined.

full_replay.py/sandbox.py (SHARED across v1-v4) were extended additively
to compute and pass this through — see their own docstrings/comments for
the exact mechanism (strategy_evolved_af's extra acq_fn(candidates) call,
sandbox.py's pool_acq_value argv slot). Confirmed backward-compatible: any
AF that doesn't read acq_value_norm is completely unaffected (defaults to
a flat, uninformative 0.5 when a caller doesn't pass it at all), and a
direct regression check showed trust_only's final_hv on a real campaign
is byte-identical before and after this change.

Everything else below (fitness metric, bootstrap CI, adaptive gamma,
checkpoint/resume, MECHANISM_FAMILIES/champion-rehash anti-mode-collapse
guards) is unchanged from evolve_af_v3.py — copied as-is.

--- evolve_af_v3.py's own docstring, kept for context (mostly still true,
    "v3"/"af_interface_v3"/"evolve_af_v3.py" self-references below refer
    to the prior file, not this one) ---

evolve_af_v3.py — copy of evolve_af_v2.py (same minimal-seed design,
af_interface_v3.py's SEED_PROGRAMS/STRATEGY_HINTS are an unmodified copy of
af_interface_v2.py's), extended ONLY to add oracle_family="tunable" (the
controlled synthetic domain in llm_af_evo/shared/tunable_synthetic_oracle.py)
to ORACLE_DEFAULTS/N_FITNESS_SEEDS_DEFAULTS below. The noise-realization
determinism gap is fixed (TunableSyntheticMOOracle.from_fixed_realization,
used by full_replay.reconstruct_oracle's "tunable" branch) and a training-
set generator exists (v3/experiments/generate_tunable_training_set.py,
one shared oracle reused across campaigns — see its own docstring for why
NOT a fresh oracle per campaign). evolve_af_v2.py itself is untouched —
this is a separate file, same pattern as v2 being separate from
evolve_af_2b.py.

Everything below this note is evolve_af_v2.py's own original docstring,
kept as-is since the fitness/design rationale it describes is unchanged:

variant of evolve_af.py using af_interface_v2.py's
minimal-seed design (see research_seed_population.md's recommendation,
written after reading FunBO/FunSearch/AlphaEvolve and the Harris-group
tuning-curve-equation-discovery paper): one full hand-written seed
(trust_only, kept only to anchor exact contract syntax/conventions) plus
one-line STRATEGY_HINTS that real-LLM mode's generation-0 bootstrap asks the
LLM to implement itself, rather than shipping 7 hand-written
implementations the way evolve_af.py/af_interface.py do.

FITNESS: full-campaign replay via full_replay.run_2b_campaign (real
sequential BO loop, evolved AF picks every batch against the real oracle),
NOT the single-step cached-pool proxy the original evolve_af.py trains on
(that "2a" proxy is invalid — see fitness_common.py's off-policy gap: a
candidate AF's per-step win/loss was scored against a front built by
EGBO-novelty's OWN trajectory, never the candidate's own hypothetical
trajectory, which is exactly what evolve_af_2b.py was already built to fix
for evolve_af.py's 7-seed design). This file is the same fitness fix
applied on top of the minimal-seed/hint/domain-generic design instead,
reusing evolve_af_2b.py's fitness approach and full_replay.py's
already-built (and, this session, stagnant_batches-corrected) replay
infrastructure directly — not reinventing full-campaign replay.

This is a SEPARATE module from evolve_af.py/evolve_af_2b.py, not a
modification of either — evolve_af.py, evolve_af_2b.py, af_interface.py,
and every existing evolution_runs/ output are left exactly as they were.
full_replay.py IS shared and was fixed in place this session (the
stagnant_batches gap — see its own module docstring's "FIXED" note); that
fix is backward-compatible (opt-in via an optional hv_history parameter)
and doesn't change evolve_af_2b.py's or any prior run's behavior.

ORACLE GENERALIZATION: full_replay.reconstruct_oracle/run_2b_campaign/
run_baseline_campaign now accept an oracle_family parameter ("excipient",
the default, or "coatings") that selects which oracle class is rebuilt
from a training log's dumped X/Y arrays — see full_replay.py's docstring.
This module's --oracle CLI flag threads that through AND derives sensible
objective_names/domain_description/feature_dim defaults for the given
family (ORACLE_DEFAULTS below), while still letting any of those be
overridden individually. A "coatings" run additionally needs training-log
JSON files in the same schema this module's load_training_campaigns
expects (X_init/Y_init/oracle_X_raw/oracle_Y_raw/budget/n_init/batch_size)
— see generate_coatings_training_set.py, which produces exactly that,
reusing the budget=40/n_init=10/batch_size=5 configuration already
validated at n=20 campaigns in coatings_generalization_results.json.

FITNESS METRIC: margin-based (mean relative final-HV improvement over the
baseline across training campaigns), not evolve_af_2b.py's binary win-rate
— switched because binary win-rate has a real resolution problem: at
n_campaigns=8 it has only 9 distinguishable values, so meaningfully
different candidates tie often; and at LARGE n_campaigns, once the
win-rate step size (1/n) shrinks below gamma*LOC's typical magnitude, the
LOC complexity penalty can start REVERSING genuine performance
differences instead of just tie-breaking between them. The margin metric
has continuous resolution at any n_campaigns and stays on a comparable
scale to the LOC penalty regardless of n. win_rate is still reported
(not used in fitness) for comparability with evolve_af_2b.py's runs.

Usage:
    python evolve_af_v4.py --train_dir ../v3/experiments/training_logs_tunable/train \\
        --oracle tunable --n_campaigns 8 --pop_size 8 --n_generations 20 \\
        --n_offspring 2 --mock --out_dir evolution_runs/run_v4

(v4 reuses v3's training_logs_tunable directly rather than duplicating it —
the logs only store oracle/init state, which full_replay.py replays fresh
every campaign regardless of the acq_value_norm change, so nothing about
the log format needed to change for this version.)
"""

import os

# MUST be set before numpy/scipy/torch are imported anywhere in this
# process — these libraries read BLAS thread-count env vars once at their
# own C-extension init time, so setting them later (e.g. via
# torch.set_num_threads after `import torch`) cannot retroactively fix an
# already-initialized BLAS backend. Same fix, same rationale, as
# run_coatings_generalization.py's identical block: that file's docstring
# documents directly verifying torch.set_num_threads(1) ALONE is
# insufficient — even the unmodified baseline strategy (no evolved-AF code
# involved) produced different final_hv across nominally-identical runs
# with only that fix in place. Neither evolve_af_2b.py nor full_replay.py
# set these (full_replay.py sets torch.set_num_threads(1) at import time,
# which is necessary but not sufficient per the above) — this file adds
# the missing piece rather than silently inheriting that same gap into a
# fresh module.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
import pathlib
import re
import sys
import traceback
import warnings

import numpy as np

_LLM_AF_EVO = pathlib.Path(__file__).resolve().parent
while _LLM_AF_EVO.name != "llm_af_evo":
    _LLM_AF_EVO = _LLM_AF_EVO.parent
_ROOT = _LLM_AF_EVO.parent
for _p in (
    _ROOT,
    _LLM_AF_EVO / "shared",
    _LLM_AF_EVO / "v1_pre_v2" / "src",
    _LLM_AF_EVO / "v1_pre_v2" / "experiments",
    _LLM_AF_EVO / "v2" / "src",
    _LLM_AF_EVO / "v2" / "experiments",
    _LLM_AF_EVO / "v3" / "src",
    _LLM_AF_EVO / "v3" / "experiments",
    _LLM_AF_EVO / "v6" / "src",
    _LLM_AF_EVO / "v6" / "experiments",
):
    sys.path.insert(0, str(_p))

from af_interface_v6 import (SEED_PROGRAMS, STRATEGY_HINTS,
                              count_loc, extract_af_docstring, has_return_statement,
                              AF_FUNCTION_NAME_V6, COMBINE_WITH_BASELINE_ACQ_V6)
from fitness_common import to_allmax
from full_replay import run_2b_campaign, run_baseline_campaign
from evaluate_multi_domain_v6 import evaluate_candidate_multi_domain_v6

HERE = pathlib.Path(__file__).parent


# ── Training data loading (campaign-level, not per-step — see module docstring) ──

def load_training_campaigns(train_dir: pathlib.Path, n: int, rng: np.random.Generator) -> list:
    """Identical to evolve_af_2b.py's load_training_campaigns."""
    files = sorted(pathlib.Path(train_dir).glob("*.json"))
    if n < len(files):
        idx = rng.choice(len(files), size=n, replace=False)
        files = [files[i] for i in sorted(idx)]
    else:
        files = files[:n]
    return [json.load(open(f)) for f in files]


# Central defaults per oracle family, so --oracle can derive sensible
# objective_names/domain_description/feature_dim/directions in one place
# instead of requiring every one of those to be specified separately for
# a coatings run. All individually overridable via their own CLI flags —
# this table only supplies the DEFAULT when the user doesn't override.
ORACLE_DEFAULTS = {
    "excipient": {
        "objective_names": ["Tm", "kD", "viscosity"],
        "domain_description": "protein formulation development",
        "feature_dim": 16,
        "directions": ["max", "max", "min"],
    },
    "coatings": {
        "objective_names": ["conductivity", "conductance_std"],
        "domain_description": "protective coatings development",
        "feature_dim": 4,
        "directions": ["max", "min"],
    },
    "tunable": {
        # See llm_af_evo/shared/tunable_synthetic_oracle.py — a controlled
        # ZDT1-family domain built specifically to avoid BOTH real domains'
        # rejected failure modes: coatings' mu_sum dominance (fixed here via
        # plateau_sharpness flattening the mean-response surface) and
        # excipient/mAb's noise floor (CV~50%; this domain's noise_level
        # defaults to 0.08 = 8% and was measured at ~1.5-2.9% CV across 20
        # campaigns in tunable_domain_generalization_results.json, roughly
        # 20-30x cleaner than excipient). NOT yet backed by a v3 training-set
        # generator — see full_replay.py's reconstruct_oracle "tunable"
        # branch and _TUNABLE_DEFAULTS comment for the remaining wiring gap
        # (noise-realization determinism across reconstruct_oracle calls).
        "objective_names": ["f1", "f2"],
        "domain_description": "a controlled synthetic multi-objective optimisation testbed",
        "feature_dim": 6,
        "directions": ["min", "min"],
    },
    "zdt1": {
        "objective_names": ["f1", "f2"],
        "domain_description": "a noisy synthetic multi-objective optimisation testbed (ZDT1, convex front)",
        "feature_dim": 6,
        "directions": ["min", "min"],
    },
    "zdt3": {
        "objective_names": ["f1", "f2"],
        "domain_description": "a noisy synthetic multi-objective optimisation testbed (ZDT3, disconnected front)",
        "feature_dim": 6,
        "directions": ["min", "min"],
    },
    "dtlz2": {
        # n_obj=3 (v6's TRAINING config — see v6/README.md). The held-out
        # n_obj=5 DTLZ2-5 domain is validation-only, evaluated outside
        # domain_configs entirely (not through this dict/the gate at all),
        # so it doesn't need its own ORACLE_DEFAULTS entry here.
        "objective_names": ["f1", "f2", "f3"],
        "domain_description": "a noisy synthetic multi-objective optimisation testbed (DTLZ2, concave front)",
        "feature_dim": 6,
        "directions": ["min", "min", "min"],
    },
}

# Adaptive-gamma ceiling: the largest LOC spread expected across a
# population, used to cap the total possible LOC penalty at one standard
# error of the fitness signal (see evaluate_af_2b's GAMMA CALIBRATION
# docstring) instead of a hand-tuned per-oracle constant. Empirically
# confirmed, not guessed: across every completed evolution run logged in
# data/evolution_runs/*/final_population.json, the observed max-min LOC
# spread within a final population ranged 5-10 (run1: 8, run2_2b: 10,
# run_v2: 7, run_v2_coatings_gamma001_v2: 8, run_v2_coatings_real[_100]: 5,
# run_v2_coatings_smoketest: 9, run_v2_mAb_real_100: 6) — 10 is the actual
# observed ceiling, not an assumption. Revisit if a future run's spread
# exceeds this.
MAX_LOC_SPREAD = 10

# Per-oracle default for how many (GP-fit, UNSGA3) seed repeats to average
# per campaign when computing fitness — see mab_noise_diagnostic.py's Axis
# 3 result: on a single fixed mAb campaign, the PAIRED relative margin
# (evolved AF vs. baseline, same joint seed) had std=0.2557 across 15
# seeds — LARGER than either raw HV series's own seed CV, and ~23x the
# ~0.011 margin gap this project is trying to detect between top AFs.
# Directly verified: pairing evolved-AF and baseline runs under a shared
# seed does NOT cancel this noise (the two strategies pick different real
# candidates from batch 1 onward, so their downstream GP-fit training data
# — and therefore their RNG streams — decorrelate almost immediately after
# the shared initial seed). Averaging fitness over several seeds per
# campaign is the cheap, direct fix for this specific noise source (no new
# real data needed, just replaying the same campaign a few extra times) —
# back-of-envelope: fully eliminating seed noise would only shrink mAb's
# implied per-campaign margin std from ~0.494 to ~0.423 (SE at n=75 from
# 0.057 to ~0.049), so this does NOT by itself solve mAb's noise-floor
# problem (domain/campaign-to-campaign variance, ~73% of the total, is
# still the dominant term and still needs the bootstrap-CI-lower-bound
# fitness and/or more training campaigns) — it is a necessary, not
# sufficient, fix.
# coatings (CV~15% cross-campaign, no seed-noise diagnostic run there
# since its signal is already well above its own noise floor) keeps the
# default at 1 (no change in behavior/cost) rather than paying 3x fitness-
# evaluation cost for a fix that domain doesn't currently need.
N_FITNESS_SEEDS_DEFAULTS = {"excipient": 3, "coatings": 1, "tunable": 1}

# Offset between repeat-seed draws for a given campaign index, large
# enough that repeat-seed values never collide with another campaign's own
# index (training sets in this repo are always far smaller than 1000
# campaigns) — see compute_baseline_hvs/evaluate_af_2b's use below.
FITNESS_SEED_STRIDE = 1000


def compute_baseline_hvs(training_logs: list, oracle_family: str = "excipient",
                          n_fitness_seeds: int = 1) -> list:
    """Identical to evolve_af_2b.py's compute_baseline_hvs — EGBO-novelty's
    real final hypervolume per training campaign, computed ONCE and shared
    across every candidate AF this run evaluates (not recomputed per
    child) — plus n_fitness_seeds > 1 support: average final_hv over
    n_fitness_seeds repeat seeds per campaign (seed = i + r*FITNESS_SEED_STRIDE
    for r in range(n_fitness_seeds)), the same seed set evaluate_af_2b uses
    below, so baseline and evolved-AF margins stay paired seed-for-seed —
    see N_FITNESS_SEEDS_DEFAULTS's docstring for why this exists.
    oracle_family is passed through to run_baseline_campaign — see
    full_replay.reconstruct_oracle's docstring."""
    print(f"Computing EGBO-novelty baseline on {len(training_logs)} training "
          f"campaigns (shared across every candidate AF this run), "
          f"averaged over {n_fitness_seeds} seed repeat(s) per campaign...")
    hvs = []
    for i, log in enumerate(training_logs):
        seeds = [i + r * FITNESS_SEED_STRIDE for r in range(n_fitness_seeds)]
        repeat_hvs = [run_baseline_campaign(log, seed=s, oracle_family=oracle_family)["final_hv"]
                      for s in seeds]
        avg_hv = float(np.mean(repeat_hvs))
        hvs.append(avg_hv)
        if n_fitness_seeds > 1:
            print(f"  campaign {i}: baseline final_hv={avg_hv:.1f} "
                  f"(mean of {repeat_hvs})")
        else:
            print(f"  campaign {i}: baseline final_hv={avg_hv:.1f}")
    return hvs


def build_pseudo_steps(training_logs: list, baseline_hvs: list,
                        directions: list = None, n_sample: int = 3) -> list:
    """
    Identical to evolve_af_2b.py's build_pseudo_steps — a lightweight
    adapter so llm_propose_child's diagnostic prompt text still shows SOME
    campaign context, without needing evolve_af.py's per-step "steps"
    shape. egbo_true_gain here is the baseline's campaign-level final_hv,
    not a per-step HV gain (different scale/meaning) — acceptable since
    this is flavour text for the prompt, not read by the fitness
    computation. directions defaults to to_allmax's own default (excipient
    Tm/kD/viscosity) for backward compatibility, but should be passed
    explicitly (ORACLE_DEFAULTS[oracle_family]["directions"]) for a
    non-excipient run — this is only used for a cosmetic front-size count
    in the prompt, but getting the sign convention wrong would silently
    miscount which points are non-dominated.
    """
    n = len(training_logs)
    idx = list(range(0, n, max(1, n // n_sample)))[:n_sample]
    steps = []
    for i in idx:
        log = training_logs[i]
        steps.append({
            "campaign": f"training_campaign_{i}", "step": log["n_init"],
            "budget": log["budget"], "stagnant_batches": 0,
            "front_allmax": to_allmax(np.array(log["Y_init"]), directions=directions),
            "egbo_true_gain": baseline_hvs[i],
        })
    return steps


# ── Fitness evaluation (full-campaign, via full_replay.run_2b_campaign) ────

def evaluate_af_2b(code: str, training_logs: list, baseline_hvs: list,
                    gamma: float = None, log_dir=None, oracle_family: str = "excipient",
                    n_fitness_seeds: int = 1) -> dict:
    """
    Full-campaign fitness — same replay mechanism as evolve_af_2b.py's
    evaluate_af_2b (oracle_family passed through to run_2b_campaign — see
    full_replay.reconstruct_oracle's docstring), plus this module's
    docstring extraction (the explainability-line feature) and MARGIN-
    BASED fitness instead of binary win-rate (see module docstring's
    FITNESS METRIC note): fitness rewards HOW MUCH a candidate beats the
    baseline by (mean relative HV improvement across training campaigns),
    not just whether it does, which avoids a discretization problem binary
    win-rate has at small n_campaigns (few distinguishable fitness values,
    ties between meaningfully different candidates) and a LOC-penalty-
    reversal problem it develops at LARGE n_campaigns (once win-rate's
    step size shrinks below gamma*LOC's typical value, small structural
    differences start dominating real performance differences). Sandbox
    failures inside a single batch are already handled gracefully by
    run_mo_campaign itself (falls back to random candidates for that
    batch) — a broken AF just produces a bad final_hv on that campaign
    rather than crashing here.

    FITNESS = a bootstrap-CI-lower-bound on the mean relative margin, minus
    an adaptively-calibrated LOC penalty:

        fitness = ci_lower_16 - gamma * loc

    This replaces the old point-estimate `margin_stat - gamma*loc` design.
    The problem that design had (directly measured on mAb/excipient):
    baseline_hv has CV~50% across 75 training campaigns, giving
    SE(mean_margin)~0.057 against top-AF margin gaps of ~0.011 (~0.2 SE) —
    the point-estimate signal there was noise, not AF quality, regardless
    of gamma, and a fixed-mean fitness has no way to tell a genuinely
    better-but-noisy AF apart from a lucky one. Using the bootstrap
    distribution's 16th percentile (approx. one SE below the mean, chosen
    over a stricter 2.5th percentile because it's less likely to discard
    genuinely good AFs from a small ~8-candidate population) instead
    directly encodes "how bad could this look under resampling noise" into
    the score AND, as a side effect, already discounts AFs whose mean is
    inflated by a handful of outlier campaigns (a real, observed case:
    gen6_child0 on coatings had mean +3.745% vs median +0.181% — a skewed
    distribution produces a wide, low bootstrap CI without needing a
    separate median/trimmed-mean step). The bootstrap always resamples
    np.mean (not a `fitness_stat` choice — median as a separate lever was
    retired, superseded by the CI-lower-bound covering the same problem).

    GAMMA CALIBRATION is now ADAPTIVE, not a hand-tuned per-oracle
    constant: gamma = SE(mean_margin) / MAX_LOC_SPREAD, where
    SE(mean_margin) = std(rel_margins) / sqrt(n_campaigns) is measured
    fresh from THIS call's own rel_margins (relative margins are
    heteroscedastic across campaigns with very different baseline_hv, so
    the noise floor is not a portable constant across datasets/domains —
    it must be measured on the actual training set being evaluated).
    MAX_LOC_SPREAD (module constant, empirically confirmed against every
    completed run's final_population.json — see its own comment) caps the
    total possible LOC penalty at exactly one SE: LOC can only break ties
    between AFs whose ci_lower values are within noise distance of each
    other, and can never override a genuine, noise-exceeding performance
    gap. Pass an explicit `gamma` to override this (e.g. for reproducing
    an old run's fixed-gamma behavior); leave it None for the adaptive
    default.

    n_fitness_seeds: number of (GP-fit, UNSGA3) seed repeats to average per
    campaign (seeds i, i+FITNESS_SEED_STRIDE, i+2*FITNESS_SEED_STRIDE, ...
    — the SAME seed set compute_baseline_hvs used for baseline_hvs[i], so
    the two stay paired). See N_FITNESS_SEEDS_DEFAULTS's docstring: this
    directly addresses seed-driven noise (mab_noise_diagnostic.py measured
    a paired-margin seed-to-seed std of 0.2557 on ONE fixed mAb campaign —
    larger than either raw HV series's own seed variance, and ~23x the
    ~0.011 margin gap this project is trying to detect), but is NOT
    sufficient on its own — domain/campaign-to-campaign noise is still the
    majority of mAb's total variance and still needs the bootstrap-CI-
    lower-bound fitness and/or more training campaigns regardless of this
    setting.

    LOGGING: writes `code` to log_dir ONCE per evaluate_af_2b call, here —
    NOT via sandbox_log_dir threaded down into run_2b_campaign/
    strategy_evolved_af (that parameter is deliberately left None below).
    Passing log_dir all the way down means run_af_in_sandbox writes a NEW
    file on every one of its calls, and strategy_evolved_af calls it ONCE
    PER BATCH — n_campaigns * batches_per_campaign (e.g. 8 * 6 = 48) writes
    of the SAME candidate's code per evaluate_af_2b call. This is exactly
    the bug evolve_af.py's own evaluate_af docstring documents fixing for
    the 2a path ("wrote the same code to a new numbered file per step...
    86,400 files from a ~17,000-call run") — carried over unfixed into the
    2b path (full_replay.py/evolve_af_2b.py never got the equivalent fix).
    """
    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(log_dir.glob("call_*.py")))
        (log_dir / f"call_{existing:05d}.py").write_text(code)

    wins, hvs, rel_margins = 0, [], []
    sig_hasher = hashlib.md5()
    for i, log in enumerate(training_logs):
        seeds = [i + r * FITNESS_SEED_STRIDE for r in range(n_fitness_seeds)]
        repeat_hvs = [run_2b_campaign(code, log, seed=s, sandbox_log_dir=None,
                                       oracle_family=oracle_family,
                                       af_function_name=AF_FUNCTION_NAME_V6,
                                       combine_with_baseline_acq=COMBINE_WITH_BASELINE_ACQ_V6)["final_hv"]
                      for s in seeds]
        hv = float(np.mean(repeat_hvs))
        hvs.append(hv)
        sig_hasher.update(str(round(hv, 2)).encode())
        baseline_hv = baseline_hvs[i]
        rel_margins.append((hv - baseline_hv) / abs(baseline_hv) if baseline_hv else 0.0)
        if hv > baseline_hv:
            wins += 1

    n = len(training_logs)
    win_rate = wins / n if n else 0.0  # retained for reporting/comparability with
                                        # evolve_af_2b.py's binary metric — NOT used
                                        # in fitness below (see module docstring's
                                        # FITNESS METRIC note)
    mean_margin = float(np.mean(rel_margins)) if rel_margins else 0.0
    median_margin = float(np.median(rel_margins)) if rel_margins else 0.0
    loc = count_loc(code, af_function_name=AF_FUNCTION_NAME_V6)

    # Bootstrap the MEAN (always — median as a separate fitness_stat lever
    # was retired, see this function's GAMMA CALIBRATION docstring above).
    # ci_lower_16 (16th percentile, ~1 SE below the mean) feeds `fitness`
    # directly; the wider 2.5th/97.5th bootstrap_ci is reporting-only, to
    # see how much of mean_margin is n_campaigns-limited uncertainty.
    ci_lower_16 = mean_margin
    bootstrap_ci = None
    if len(rel_margins) >= 2:
        boot_rng = np.random.default_rng(0)
        boot_samples = [
            float(np.mean(boot_rng.choice(rel_margins, size=len(rel_margins), replace=True)))
            for _ in range(2000)
        ]
        ci_lower_16 = float(np.percentile(boot_samples, 16))
        bootstrap_ci = [float(np.percentile(boot_samples, 2.5)),
                         float(np.percentile(boot_samples, 97.5))]

    # v5 addition: computed UNCONDITIONALLY now, not just when gamma is
    # None. Previously this was only computed inside adaptive-gamma's own
    # branch — harmless for v4 (gamma is usually left None there), but v5
    # always passes gamma=0 explicitly (see evaluate_af_multi_domain), so
    # se silently never got computed at all. Needed now as the per-domain
    # precision input to combine_domain_scores' inverse-variance
    # weighting (see multi_domain_fitness.py) — a candidate's se here is
    # exactly what SE(ci_lower_16) requires to know how much to trust
    # this domain's estimate relative to the others being combined.
    se_mean_margin = float(np.std(rel_margins, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if gamma is None:
        gamma = se_mean_margin / MAX_LOC_SPREAD

    fitness = ci_lower_16 - gamma * loc

    return {"win_rate": win_rate, "mean_margin": mean_margin,
            "median_margin": median_margin, "ci_lower_16": ci_lower_16,
            "bootstrap_ci": bootstrap_ci, "gamma": gamma,
            "se_mean_margin": se_mean_margin,
            "loc": loc, "fitness": fitness,
            "mean_hv": float(np.mean(hvs)) if hvs else float("nan"),
            "n_campaigns": n, "selection_signature": sig_hasher.hexdigest(),
            "docstring": extract_af_docstring(code, af_function_name=AF_FUNCTION_NAME_V6)}


# ── Multi-domain fitness (v5 addition) ──────────────────────────────────────
#
# See the af-evolution branch conversation's multi-domain-fitness design
# discussion for the full reasoning behind every choice below — this is
# the implementation of that design, wired to evaluate_af_2b (the
# per-domain primitive, unchanged above) via evaluate_multi_domain.py's
# gating/combination logic (multi_domain_fitness.py), both verified
# standalone against real training data before this integration.
#
# Result-dict field mapping vs evaluate_af_2b's single-domain shape,
# deliberately kept where the meaning still makes sense so downstream
# code (llm_propose_child's prompt text, the anti-mode-collapse guards,
# checkpoint save/load) needs NO changes at all:
#   fitness            unchanged meaning: what population sorting uses.
#   mean_margin        REPURPOSED: the combined (mean-lam*std) score
#                       across domains that passed their gates, BEFORE the
#                       gamma*loc penalty — analogous to "how good is this
#                       candidate, before the complexity penalty", same as
#                       the field's old single-domain meaning, just now a
#                       cross-domain combination instead of one domain's
#                       margin. Still a percentage-like fraction, so
#                       llm_propose_child's `{value:+.1%}` prompt
#                       formatting keeps working unchanged. GATE_FAIL_
#                       FITNESS-scale (very negative) for a gated-out
#                       candidate — never meant to be read as a real
#                       margin in that case, only compared for ranking.
#   win_rate            REPURPOSED: fraction of domains this candidate
#                       actually passed the gate on (e.g. 2/3 if it
#                       cleared tunable+coatings but failed mAb) — the
#                       genuinely multi-domain-relevant analog of "how
#                       often did this beat the baseline", since "beating
#                       the baseline on one campaign" doesn't mean
#                       anything cross-domain the way it did within one
#                       domain.
#   loc, docstring,     unchanged meaning, computed from `code` exactly as
#   selection_signature before — anti-mode-collapse guards and jitter's regex
#                       operate on these directly and need no changes.
# Dropped (no honest multi-domain analog, NOT force-fit into old meaning):
#   ci_lower_16, bootstrap_ci, median_margin, gamma (per-call), mean_hv,
#   n_campaigns — mean_hv in particular would average HV values across
#   domains with wildly different absolute scales (tunable ~11,
#   coatings ~400, mAb ~9000), which is meaningless; per-domain HVs live
#   in the new `domain_scores` field instead (see below).
# New fields:
#   domain_scores       {domain_name: ci_lower_16} for every domain
#                       ACTUALLY evaluated — partial (fewer entries than
#                       the full domain list) if gated out early, since
#                       later domains are never called (that's the
#                       funnel's whole cost-saving point).
#   gate_failed_at      the domain name that failed the gate, or None if
#                       every domain passed.
GAMMA_V6_DEFAULT = 0  # unchanged rationale from v5: start at 0, recalibrate
                       # only if complexity-driven formulas start winning —
                       # applied ONCE to the final combined fitness (see
                       # below), never per-domain.


def evaluate_af_multi_domain(code: str, domain_configs: dict, gamma: float = GAMMA_V6_DEFAULT,
                              log_dir=None, gate_z_threshold=None) -> dict:
    """Multi-domain analog of evaluate_af_2b above — v6 version, using
    multi_domain_fitness_v6's z-scored-mean-no-std combination (see that
    module's docstring for why v5's mean-lam*std was dropped) instead of
    v5's combine_domain_scores_weighted. domain_configs: ORDERED
    {domain_name: {"training_logs", "baseline_hvs", "oracle_family",
    "n_fitness_seeds"}} — order IS the gate sequence (v6: ZDT1 -> DTLZ2-3
    -> ZDT3, ascending noise, cheapest/cleanest-read first — see
    v6/README.md)."""
    combined_raw, domain_scores, domain_zs, gate_failed_at = evaluate_candidate_multi_domain_v6(
        code, domain_configs, evaluate_af_2b, log_dir=log_dir,
        gate_z_threshold=gate_z_threshold)
    loc = count_loc(code, af_function_name=AF_FUNCTION_NAME_V6)
    if gate_failed_at is None:
        fitness = combined_raw - gamma * loc
        mean_margin = combined_raw
    else:
        # A gated-out candidate's fitness is already graded (see
        # multi_domain_fitness_v6.GATE_FAIL_BASE/DOMAIN_PASS_BONUS) far
        # below anything a passing candidate can score — the complexity
        # penalty is irrelevant to its ranking, so it isn't applied here.
        fitness = combined_raw
        mean_margin = combined_raw
    win_rate = len(domain_scores) / len(domain_configs) if gate_failed_at is not None \
        else 1.0  # fraction of domains passed — see header comment
    sig_hasher = hashlib.md5()
    for name in domain_configs:  # fixed iteration order (dict insertion
                                   # order == gate sequence), not
                                   # domain_scores' possibly-partial keys,
                                   # so two candidates gated out at
                                   # different domains still hash
                                   # differently even if their scores so
                                   # far happen to match by coincidence
        sig_hasher.update(str(round(domain_scores.get(name, float("nan")), 4)).encode())
    return {"fitness": fitness, "mean_margin": mean_margin, "win_rate": win_rate,
            "domain_scores": domain_scores, "domain_zs": domain_zs,
            "gate_failed_at": gate_failed_at,
            "loc": loc, "selection_signature": sig_hasher.hexdigest(),
            "docstring": extract_af_docstring(code, af_function_name=AF_FUNCTION_NAME_V6)}


# ── Real-LLM child generation (wired, untestable in this sandbox) ─────────

def _build_system_prompt(objective_names: list, domain_description: str,
                          feature_dim: int = 16) -> str:
    """
    Builds the system prompt generically over objective_names/domain_
    description/feature_dim, rather than hardcoding "protein formulations",
    Tm/kD/viscosity, and 16 input dimensions the way evolve_af.py's
    _LLM_SYSTEM_PROMPT constant does. feature_dim defaults to 16 (the
    excipient oracle's dimensionality) for backward compatibility;
    coatings is 4D (ada_coatings_oracle.FEATURE_DIM) — getting this wrong
    wouldn't break anything mechanically (candidates still carry the real
    x array regardless of what the prompt claims its shape is), but it
    would describe the contract incorrectly to the LLM, which is exactly
    the kind of domain-mismatch this whole prompt-generalization effort is
    about removing.
    """
    # v5 addition: feature_dim=None (the multi-domain default — see
    # run_evolution's _MULTI_DOMAIN_DESCRIPTION comment) must NOT render
    # as literal "None" inside a shape position like np.ndarray(None,) —
    # confusing, reads as a real (wrong) type hint rather than "varies".
    # "d" is standard math notation for an unspecified dimension count and
    # reads correctly in context either way.
    feature_dim_text = feature_dim if feature_dim is not None else "d"
    names = list(objective_names)
    gp_lines = "\n".join(
        f'             "{n}": {{"mean": float, "std": float}},  # higher is better, already flipped if needed'
        for n in names)
    range_dict = ", ".join(f'"{n}": float' for n in names)
    # acq_value_norm: v4 addition, see sandbox.py's pool_acq_value docs.
    acq_line = (
        '         "acq_value_norm": float,  # botorch\'s real qLogNEHVI acquisition value for this\n'
        '                                    # candidate (Monte-Carlo integrated hypervolume-improvement\n'
        '                                    # estimate over the FULL joint objective posterior), min-max\n'
        '                                    # normalised to [0,1] across the whole pool')

    return f'''You are an expert in {domain_description} and in Bayesian \
optimisation, evolving a CORRECTION TERM added to a real acquisition \
function for a multi-objective Bayesian optimisation loop in this domain. \
You must write a Python function:

def modifier(context) -> list:
    ...
    return values  # one value per entry in context["pool"] — a CORRECTION \
TERM, NOT a full score (see below)

CRITICAL — READ THIS BEFORE ANYTHING ELSE: your modifier() function does \
NOT compute the final score. The final score used to pick each batch is \
computed AUTOMATICALLY, OUTSIDE your code, as:

    final_score[i] = context["pool"][i]["acq_value_norm"] + modifier(context)[i]

acq_value_norm is botorch's real qLogNEHVI acquisition value (see below) \
— it is ALREADY the dominant, high-quality term. Your modifier() function \
must return ONLY the correction to ADD on top of it — a bonus or penalty, \
NOT a rewrite of the whole score. Do NOT read acq_value_norm and add it \
into your own return value — it is added for you automatically, EXACTLY \
ONCE, after your function returns. If you add it yourself too, it gets \
counted TWICE, which silently breaks the acquisition function. You MAY \
read context["pool"][i]["acq_value_norm"] if you want your correction to \
scale with it (e.g. only apply a bonus to candidates that are already \
acquisition-strong) — reading it is fine, returning it (or adding it into \
what you return) is not.

A modifier that returns all zeros is equivalent to the raw acq_value_norm \
baseline with no correction at all — that is your fallback/floor, not \
your goal. Your job is to find a correction (an uncertainty bonus, a \
novelty term, a front-coverage-gap bonus, a stagnation-adaptive boost, \
etc. — see the strategy hint you were given, if any) that measurably \
improves on that floor.

REQUIRED — the first statement in your function body must be a one-line \
docstring in plain English describing what your CORRECTION does, e.g.:

def modifier(context):
    """<one line: what correction this adds on top of acq_value_norm>"""
    ...

This is not optional decoration: modifier programs that don't start with \
a non-empty one-line docstring are rejected outright before ever being run \
or scored. The sentence should describe the CORRECTION's strategy (e.g. \
"add an uncertainty bonus that decays as the campaign progresses"), not \
restate the code line-by-line, and should NOT describe acq_value_norm \
itself (that part is fixed, not something you're writing).

context = {{
    "pool": [
        {{"x": np.ndarray({feature_dim_text},),               # normalised candidate features
         "gp_posterior": {{
{gp_lines}
         }},
{acq_line}
        }},
        ...  # one entry per candidate
    ],
    "X_obs": np.ndarray(n_obs, {feature_dim_text}),          # every candidate observed so far
    "Y_obs": np.ndarray(n_obs, {len(names)}),  # every OUTCOME observed so far, all-maximised,
                                              # rows match X_obs, columns match objective_names
                                              # (the full observation history, not filtered to
                                              # the front — use for novelty distance or
                                              # resampling a noisy Pareto front)
    "objective_names": {names!r},   # column order for the array forms below
    "pareto_front": np.ndarray(n_pf, {len(names)}),  # non-dominated points, columns match objective_names
    "pareto_front_range": {{{range_dict}}},  # observed range per objective
    "ref_point": np.ndarray({len(names)},),  # HV reference point, same column order as objective_names
    "ref_point_by_name": {{{range_dict}}},
    "campaign": {{"step": int, "budget": int, "progress": float,  # step/budget, in [0,1]
                 "n_obs": int, "stagnant_batches": int}},  # consecutive non-improving batches
}}

Every objective in context["gp_posterior"] is ALREADY oriented so higher is always \
better — do not flip any of them again. Access objectives BY NAME (e.g. \
cand["gp_posterior"]["{names[-1]}"]["std"]), never by a remembered array position, \
and iterate context["objective_names"] rather than hardcoding this run's specific \
objective set, since the same modifier contract is reused across different domains.

THE BASELINE YOU ARE TRYING TO IMPROVE ON: the current production method
("EGBO-novelty") does NOT use a linear combination of mean and std. Each
batch it: (1) fits a GP per objective, (2) proposes candidates by
optimising an acquisition score that estimates how much a candidate would
expand the dominated hypervolume, integrated over the GP's posterior
uncertainty (Monte Carlo sampled), (3) generates more candidates via an
evolutionary algorithm seeded from the current Pareto front, (4) scores
the combined pool with that same hypervolume-improvement estimate, then
(5) picks a batch using a rule that also rewards candidates far from
previously-observed points (novelty), not just high acquisition value.

context["pool"][i]["acq_value_norm"] IS step (4)'s exact real signal — the
same hypervolume-improvement estimate the baseline computes, MC-integrated
over the full joint objective posterior, min-max normalised across the
pool — and it is ALREADY being used as the base score, automatically,
outside your code (see above). You are being asked for step (5): a
correction on top of it. Do NOT spend effort hand-deriving your own
hypervolume-improvement proxy from gp_posterior mean/std (e.g. resampling
candidate posteriors, or estimating each candidate's probability of being
Pareto-optimal by Monte Carlo sampling) — acq_value_norm already IS that
computation, done properly, so re-deriving any part of it by hand can only
ever produce a strictly worse approximation. Focus entirely on the
CORRECTION: what should shift a candidate's rank up or down relative to
its raw acquisition value, and by how much.

WHAT pareto_front AND ref_point ARE FOR: they are provided so you can
reason about hypervolume improvement directly, if your correction wants to
use front geometry — how much a candidate would expand the region of
objective space that's better than every current non-dominated point, if
its predicted objectives are correct. A candidate whose predicted
objectives fall outside (beyond) the current front in some direction, and
far from where the front already reaches, expands the dominated region
more than one that's near the front or dominated by it. front_range gives
the current front's per-objective spread if you want to normalise. You are
not required to use these fields.

WHAT YOUR SCORE IS MEASURED BY: your fitness is the MEAN RELATIVE
IMPROVEMENT in final hypervolume across FULL TRAINING CAMPAIGNS (not
individual decision points, and not just a win/loss count) — using
acq_value_norm + your modifier as the acquisition function for every batch
of a real sequential run, how much better (or worse) is the final
hypervolume than EGBO-novelty's own real final hypervolume on that same
campaign, averaged across campaigns. fitness = mean_relative_improvement -
(a small penalty per line of code). This means a choice that looks good
for one batch but leaves LATER batches worse off (e.g. by never exploring)
is punished directly by this fitness, not just caught by a separate later
check — consider how your choices at one decision compound into later
ones, not just whether this one decision looks good in isolation. It also
means margin matters: beating the baseline by a wide margin scores
meaningfully better than barely beating it, unlike a simple win/loss count.

You're trained across SEVERAL domains at once (see above), so the number
shown to you below as "avg_z" is a cross-domain average, not a single
percentage margin: each domain's relative improvement is divided by its
own measurement uncertainty (so domains with noisier signal don't
dominate purely from being noisier), then those per-domain values are
averaged. avg_z=0 means "matches baseline, averaged across domains it was
tested on"; positive means genuinely better, adjusted for measurement
noise; roughly +1 or more is a solid, trustworthy improvement, not just
noise. It is NOT a percentage — do not describe it as one.

HOW YOUR VALUES ARE USED: after modifier() returns, the sandbox adds
acq_value_norm to each value (exactly once — see above), and the top
batch_size candidates BY THAT COMBINED SCORE are taken directly as the
batch — there is no additional filtering, diversity, or hypervolume-
maximisation step after that. This means: if you want the batch to be
diverse, your correction itself has to produce that (e.g. via a novelty
term) — nothing downstream will do it for you.

Worked example — a simple uncertainty-bonus modifier:

def modifier(context):
    """Uncertainty bonus (UCB-style), decaying as the campaign progresses."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    weight = 0.3 * (1.0 - context["campaign"]["progress"])
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        values.append(weight * sigma_norm)
    return values

This ADDS a bonus for candidates the GP is still uncertain about (an
exploration nudge on top of acq_value_norm's exploitation signal), heavier
early in the campaign and fading toward zero as it progresses — it does
NOT recompute mu_sum or anything acq_value_norm already covers. You can do
better: e.g. use context["X_obs"] to add a novelty term (distance from
cand["x"] to the nearest observed point), or scale your correction by
acq_value_norm itself (read it, don't return it) so it only applies to
already-strong candidates.

IMPORTANT — a mistake that silently does nothing: context["campaign"]'s
fields (step, budget, progress, stagnant_batches) are IDENTICAL for every
candidate in the pool — they do not vary across cand in context["pool"].
Because scores only matter through their RELATIVE ORDER (the top-scoring
candidates get picked, not the absolute score values), multiplying every
candidate's score by the same campaign-level number changes NOTHING about
which candidates are selected. For example `k(progress) * sigma_norm(cand)`
alone, for ANY function k of only progress/stagnant_batches, always picks
the exact same candidates as plain `sigma_norm(cand)` — the phase-awareness
is completely inert. campaign state only changes what gets picked when it
shifts the BALANCE between two or more candidate-varying terms (like
mu_sum and sigma_norm), e.g. `w(progress) * sigma_norm(cand) + (1 -
w(progress)) * mu_sum(cand)` — never use it as a lone multiplier on a
single term, and do not drop mu_sum entirely (an AF with no exploitation
signal at all cannot compete with a method that balances both).

Only numpy is available (import numpy as np already provided). Return ONLY \
the Python function, no markdown fences, no explanation. Prefer SIMPLER \
expressions — the fitness function penalises longer code, so a shorter \
function that performs nearly as well beats a longer one that performs \
slightly better.'''


def _campaign_diagnostic_text(steps: list, n_campaigns: int = 3) -> str:
    """Text/numeric stand-in for the Harris group's graphical diagnostics
    — see build_pseudo_steps for how these pseudo-step records are built
    from campaign-level (not per-step) training data."""
    sample = steps[:: max(1, len(steps) // n_campaigns)][:n_campaigns]
    lines = []
    for s in sample:
        lines.append(f"  {s['campaign']} step {s['step']}/{s['budget']} "
                      f"(stagnant_batches={s['stagnant_batches']}): front_size="
                      f"{len(s['front_allmax'])}, EGBO-novelty final HV="
                      f"{s['egbo_true_gain']:.2f}")
    return "\n".join(lines)


def llm_generate_seed_from_hint(hint: str, model: str, rng: np.random.Generator,
                                 objective_names: list, domain_description: str,
                                 feature_dim: int = 16) -> str:
    """
    Generation-0 bootstrap for real-LLM mode: ask the LLM to WRITE
    modifier implementing a one-line STRATEGY_HINT, rather than shipping
    a hand-written implementation for every strategy. Same fence-stripping/
    error-propagation contract as llm_propose_child; callers should catch
    exceptions the same way (see run_evolution's population-init loop).

    Retries ONCE, with a stronger anti-repetition configuration, if the
    first attempt never reaches a `return` statement (af_interface.
    has_return_statement) — this is the repetition-loop pathology found on
    noisy_front_hvi (run_v2_coatings_gamma001_v2): the model spiraled
    through ~15 near-duplicate hedging comments ("this is not a correct
    implementation... a full implementation would...") and burned the
    entire num_predict budget before writing any code, despite the prompt
    already asking for at most one comment per step. A higher
    repeat_penalty and lower temperature target the repetition-loop
    specifically (as opposed to num_predict, which a pure length increase
    would not fix — the failure is qualitative, not "ran out of room").
    Falls through and returns the second attempt's code regardless of
    whether IT has a return either — the sandbox's own missing-return
    rejection is still the final backstop, this is just a cheap
    first-chance retry before paying that cost.
    """
    import ollama

    system_prompt = _build_system_prompt(objective_names, domain_description, feature_dim)
    prompt = (
        f"Implement the following acquisition strategy as a modifier "
        f"(a correction term added to acq_value_norm automatically — see "
        f"the contract above; do NOT return acq_value_norm yourself):\n\n"
        f"\"{hint}\"\n\n"
        f"Write a complete, correct modifier(context) function for exactly "
        f"this strategy. Do not add features beyond what's described above "
        f"(no extra terms, no phase-awareness, no novelty) unless the "
        f"description explicitly calls for it — this seed's job is to "
        f"implement ONE described strategy faithfully, not to be creative.\n\n"
        f"Decide your approach ONCE, silently, before writing anything. Then "
        f"write the code directly. Do not think out loud in comments, do "
        f"not narrate alternative approaches or reconsider your approach "
        f"mid-function, and do not repeat the same caveat/disclaimer in "
        f"multiple comments — at most one short comment per logical step. "
        f"A function that is simpler than the ideal but complete and "
        f"correct beats one that is more sophisticated but truncated or "
        f"never finished."
    )

    def _generate(temperature: float, repeat_penalty: float, extra_prompt: str = "") -> str:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt + extra_prompt}],
            options={"temperature": temperature, "num_predict": 1024,
                     "repeat_penalty": repeat_penalty,
                     "num_ctx": 8192,
                     "seed": int(rng.integers(1_000_000))},
            # Keep the model resident between calls: this run interleaves
            # LLM calls with full_replay campaign evaluation (GP fits,
            # optimize_acqf) that routinely exceeds Ollama's 5-minute
            # default keep_alive, which would otherwise force a disk
            # reload (multi-GB) on every subsequent call.
            keep_alive="30m",
        )
        code = resp["message"]["content"].strip()
        for fence in ["```python", "```"]:
            if code.startswith(fence):
                code = code[len(fence):]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    code = _generate(temperature=0.3, repeat_penalty=1.3)
    if not has_return_statement(code, af_function_name=AF_FUNCTION_NAME_V6):
        code = _generate(
            temperature=0.1, repeat_penalty=1.6,
            extra_prompt=(
                "\n\nSTOP: write the return statement as your very next "
                "line of code if you have not already. Do not write any "
                "more comments, caveats, or explanations before it — "
                "code first, one line of comment maximum for the whole "
                "function."
            ),
        )
    return code


# ── Anti-mode-collapse: stagnation-mechanism rotation ───────────────────────
#
# Found on the tunable-domain run1 (see run1's history.json/af_code_logs):
# 25 of 30 generations were stagnant, and every one of those ~50 LLM calls
# proposed near-verbatim restatements of ONE idea — "resample the GP
# posterior under noise and count dominance/HV-expansion frequency" — which
# is literally one of the annealing note's own suggested examples below. The
# model anchored on that one exemplar and never tried the others as a
# PRIMARY mechanism, despite the note's instruction to pick a structurally
# different one each time. This registry + classify_mechanism_families let
# the loop notice that anchoring is happening (by classifying each
# stagnation-triggered child's docstring) and explicitly forbid/deprioritize
# whichever family has already been retried, forcing rotation instead of
# repetition.
MECHANISM_FAMILIES = [
    ("diversity_repulsion",
     "diversity/repulsion between top-ranked candidates",
     ["repuls", "diversity", "repel", "spread out", "penal", "crowd"]),
    ("proximity_suppression",
     "greedy proximity-based suppression of near-duplicates",
     ["proximity", "suppress", "greedy", "near-duplicate", "near duplicate",
      "nearest neighbor", "iterative"]),
    ("posterior_resampling",
     "resampling the observed history/candidates' posteriors under noise "
     "to estimate improvement or dominance",
     ["resampl", "monte carlo", "mc sampl", "sampled objective",
      "sampling from", "under noise"]),
    ("pareto_probability",
     "estimating each candidate's probability of being Pareto-optimal or "
     "of expanding the hypervolume",
     ["pareto-optim", "pareto optim", "dominance probability",
      "hypervolume improvement", "hypervolume expansion", "expand hypervolume",
      "dominate", "non-dominat"]),
]
# A family counts as "worn out" this stagnation streak once it's been the
# dominant idea in this many stagnation-triggered children without
# producing an improvement — 2 lets the model try a family once, retry it
# once (in case the first attempt was just a bad implementation), then
# forces a switch.
FAMILY_REPEAT_LIMIT = 2


def classify_mechanism_families(docstring: str) -> list:
    """Best-effort keyword classification of which MECHANISM_FAMILIES a
    child's one-line docstring belongs to (can match more than one, e.g. a
    child that blends resampling with a repulsion term — see run1's
    gen8_child1). Returns [] for anything that doesn't match a listed
    family (e.g. plain UCB/novelty variants), which is fine: this registry
    only exists to stop repetition of the specific families the annealing
    note suggests, not to classify every possible idea."""
    text = (docstring or "").lower()
    return [name for name, _desc, keywords in MECHANISM_FAMILIES
            if any(kw in text for kw in keywords)]


# ── Anti-mode-collapse, part 2: rehashing the incumbent champion ───────────
#
# Found on run2 (the first run with the MECHANISM_FAMILIES fix above): the
# fix worked exactly as designed — family_attempt_counts showed genuine
# rotation through all four suggested families (diversity_repulsion=7,
# pareto_probability=5, proximity_suppression=4, posterior_resampling=2)
# instead of run1's ~90%-one-family collapse — but ALL FOUR were tried,
# evaluated, and correctly rejected (none survived into the final
# population), and the run still went on to stagnate for 30 straight
# generations anyway. Reading the docstrings showed why: once every
# suggested family was "worn out", the fallback instruction ("compose two
# together or invent something new") gave the model room to just keep
# rewording the CURRENT CHAMPION'S OWN idea instead — the entire final
# population (gen12 through gen47) is cosmetic restatements of one
# "progress-adaptive exploitation + uncertainty scaling + hypervolume
# normalization" formula, dressed in different vocabulary each time
# ("dynamic reward shaping", "sigmoidal blending", "exponential decay").
# That's literally the "small weight variation" behavior the annealing
# note already tells the model not to do — MECHANISM_FAMILIES just never
# measured it because it isn't one of the four suggested exemplars, it's
# a moving target (whatever the current best_so_far happens to be).
#
# CHAMPION_REHASH_NAME is tracked the same way as MECHANISM_FAMILIES
# entries (counted per stagnation streak, reset on improvement, persisted
# through checkpoint/resume) but classified differently: not by fixed
# keywords, but by word-overlap similarity to best_so_far's OWN docstring
# at proposal time. Threshold picked by direct measurement against run2's
# actual docstrings: genuine rehashes of run2's champion scored 0.36-0.64,
# genuinely different ideas scored 0.00-0.07 — 0.3 sits cleanly in the gap.
CHAMPION_REHASH_NAME = "champion_rehash"
CHAMPION_REHASH_THRESHOLD = 0.3
_REHASH_STOPWORDS = {
    "with", "and", "the", "based", "using", "from", "that", "this", "their",
    "then", "than", "into", "onto", "over", "under", "each", "some", "more",
    "less", "most", "also", "such", "like", "when", "while", "where", "does",
    "doesnt", "via", "per", "for", "not", "are", "has", "have", "can", "will",
    "all", "any", "own", "due",
}


def _rehash_words(docstring: str) -> set:
    words = re.findall(r"[a-z]+", (docstring or "").lower())
    return {w for w in words if w not in _REHASH_STOPWORDS and len(w) > 3}


def champion_rehash_similarity(child_docstring: str, champion_docstring: str) -> float:
    """Jaccard similarity of meaningful words between a child's docstring
    and the current champion's — see CHAMPION_REHASH_NAME's note above.
    0.0 if either docstring is missing/has no meaningful words (never
    falsely flags a rehash from empty input). Generic pairwise-docstring
    similarity — reused as-is by generic_repeat_similarity below to
    compare a child against non-champion siblings too."""
    words_a = _rehash_words(child_docstring)
    words_b = _rehash_words(champion_docstring)
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


# ── Anti-mode-collapse, part 3: rehashing a non-champion idea ──────────────
#
# Found on v4 run1: with parts 1+2 above both active and working as
# designed (family_attempt_counts showed genuine rotation through all four
# MECHANISM_FAMILIES, plus champion_rehash correctly climbing and
# triggering the hard-redirect), the run STILL stagnated for 16+ straight
# generations. Reading the actual generated code showed why: calls 37, 38,
# and 43 were three cosmetic restatements of "UCB-style score + inverse-
# distance novelty" (a formula that was never the champion — the champion
# stayed the gen-0 acq_value_norm-based seed throughout), and calls 40/41
# were BYTE-IDENTICAL. None of this got caught, because
# classify_mechanism_families' keyword lists don't cover "novelty"/"UCB"
# phrasing, and champion_rehash_similarity only ever compares a child
# against best_so_far — a repeated idea that ISN'T the champion is
# invisible to both mechanisms. The 4 named families + the champion aren't
# an exhaustive list of what the model can fixate on; they're just the two
# failure modes we'd previously observed and named. This third guard
# generalizes the fix: instead of comparing against one fixed reference
# (the champion), compare each new child against a rolling window of ALL
# recent stagnation-triggered children's docstrings, regardless of which
# family (if any) they were classified into. Same word-overlap similarity
# function and 0.3 threshold as champion_rehash — already validated
# against real docstrings in this run's own logs (see above).
GENERIC_REPEAT_NAME = "generic_repeat"
RECENT_DOCSTRING_WINDOW = 15


def generic_repeat_similarity(child_docstring: str, recent_docstrings: list) -> float:
    """Max champion_rehash_similarity between child_docstring and any entry
    in recent_docstrings (a rolling window of recently-proposed children's
    docstrings, not just the champion's). 0.0 if recent_docstrings is
    empty."""
    if not recent_docstrings:
        return 0.0
    return max(champion_rehash_similarity(child_docstring, d) for d in recent_docstrings)


# Static family tags for STRATEGY_HINTS, used only by unexhausted_strategy_hint
# below. classify_mechanism_families (keyword-scanning a free-form LLM-
# written docstring) is the wrong tool here: STRATEGY_HINTS' descriptions
# are our own fixed, hand-written text, so what family each one belongs to
# is known outright rather than needing to be guessed from keywords — and
# guessing gets it wrong for exactly the hints that matter most. Verified
# directly: classify_mechanism_families("Score by predicted objective sum
# plus a fixed-weight (beta=2.0) uncertainty bonus, UCB-style...") returns
# [] (no MECHANISM_FAMILIES keyword — "resampl", "pareto-optim" etc. — asks
# about UCB at all), so fixed_ucb slipped straight through a keyword-only
# filter in testing.
#
# v4.1: af_interface_v4.py's STRATEGY_HINTS dropped fixed_ucb/
# ucb_plus_novelty/phase_decaying_ucb (see its own comment for the full
# account — every run's champion so far has converged to the SAME family
# these three hints describe, so offering them as a stagnation "escape"
# wasn't actually offering anywhere new to go) in favour of
# front_coverage_gap/obj_correlation_bonus/improvement_momentum, none of
# which carry this tag. "ucb_uncertainty" is kept as a tag (now naming
# only acq_value_progress_blend) rather than removed outright: that hint
# is still the literal formula every champion has converged to
# (acq_value_progress_blend seeded egbo_novelty_like, the seed every real
# champion so far has been a variant of), so it still needs unconditional
# exclusion from the redirect fallback even though its three siblings are
# gone. If a future STRATEGY_HINTS revision adds another hint that's
# mechanically UCB+uncertainty-shaped, tag it here too.
#
# (Two text-similarity-based approaches to detecting "this hint's already
# been reworded" — Jaccard, then an overlap coefficient — were tried and
# discarded here; see unexhausted_strategy_hint's docstring below for why
# neither held up against real data, and why exact per-key tracking via
# family_attempt_counts replaced them instead.)
_UCB_UNCERTAINTY_TAG = "ucb_uncertainty"
STRATEGY_HINT_TAGS = {
    "acq_value_progress_blend": {_UCB_UNCERTAINTY_TAG},
    "noisy_front_hvi": {"posterior_resampling"},
    "outcome_novelty": set(),
    "dpp_diversity": {"diversity_repulsion"},
    "local_penalization": {"proximity_suppression"},
    "pareto_membership": {"pareto_probability"},
    "dro_robust_hvi": {"posterior_resampling"},
    "front_coverage_gap": set(),
    "obj_correlation_bonus": set(),
    "improvement_momentum": set(),
}


def unexhausted_strategy_hint(rng, family_attempt_counts: dict, worn_out_families: set,
                               best_doc: str) -> tuple:
    """Pick a (key, description) pair from STRATEGY_HINTS for the
    hard-redirect fallback, filtered against what's already been exhausted
    this stagnation streak. The caller MUST tally the returned key into
    family_attempt_counts (as f"hint_{key}") immediately after using it —
    see HINT_KEY_PREFIX below — or this function has no way to know a hint
    got reused and will keep suggesting it.

    Found needed on v4 run1's second stagnation stretch (gens 22-41): once
    all four MECHANISM_FAMILIES were worn out, the hard-redirect fallback
    used to do `random.choice(list(STRATEGY_HINTS.values()))` — completely
    unfiltered. STRATEGY_HINTS is the *gen-0 seed pool*, not a curated
    "still fresh" list like MECHANISM_FAMILIES is — at the time this was
    found, it held 9 hints including fixed_ucb/ucb_plus_novelty/
    phase_decaying_ucb/acq_value_progress_blend, all UCB+uncertainty
    (+novelty) variants (the first three have since been dropped from the
    catalog entirely — see STRATEGY_HINT_TAGS' v4.1 comment above — but
    this function's job, filtering ANY future catalog against what's
    already worn out, doesn't depend on which specific hints happen to be
    in it). So the redirect could (and on run1 did) hand the model back
    "Combine a UCB-style exploration credit... with an explicit novelty
    term" — textually different from, but
    conceptually identical to, the champion's own acq_value+uncertainty
    formula the model was already stuck rewording.

    Two iterations before this one both tried to detect "has this hint
    already been suggested and reworded" by comparing STRATEGY_HINTS' own
    (long, detailed) description text against recent_docstrings (short,
    one-line child summaries) via word-overlap similarity — first Jaccard
    (champion_rehash_similarity's metric), then an overlap coefficient
    (_hint_similarity). Both measured directly against run1's actual gens
    41-45 stuck state: Jaccard scored real restatements of outcome_novelty
    at only 0.12-0.15 (swamped by the long hint's own vocabulary — false
    negative, let the same hint be resuggested 5 times), and the overlap
    coefficient over-corrected the other way — taking the max similarity
    against a 15-entry recent_docstrings window flagged EVERY hint as "too
    similar to something recent" (0.33-1.0 across the board) purely from
    generic domain vocabulary shared by any two AF ideas ("candidates",
    "objective", "predicted", "score") — false positive, starved the pool
    down to a single tag-surviving hint regardless of what was actually
    repeating. Comparing a fixed catalog entry's prose against a
    stagnation-window's worth of free-form LLM summaries is the wrong tool
    either way — thresholds calibrated on a couple of examples don't hold
    up against real, noisy, many-comparison data.

    What actually distinguishes "this hint was suggested and failed" is
    not text similarity at all — it's whether THIS EXACT KEY has already
    been handed to the redirect prompt this streak, which the caller knows
    exactly (it's the return value of this same function, previous calls).
    So family_attempt_counts gets a new per-hint counter
    (f"hint_{key}", same reset-on-improvement lifecycle as every other
    entry in that dict) instead of trying to infer reuse from prose.

    Filters, most-constrained first, each only advancing to the next if it
    empties the candidate pool entirely (never returns nothing to try):
    1. tag (STRATEGY_HINT_TAGS not in an already-worn-out MECHANISM_FAMILIES
       family or the champion's own ucb_uncertainty family) AND not itself
       already redirected-to >= FAMILY_REPEAT_LIMIT times this streak.
    2. Drop the tag constraint, keep "not already worn out by key".
    3. Drop the key-repeat constraint too (last resort — every hint has
       been tried and tag-worn-out; just avoid the champion's own family).
    4. Give up: any hint at all.
    """
    hint_worn_out = {key for key in STRATEGY_HINTS
                      if family_attempt_counts.get(f"hint_{key}", 0) >= FAMILY_REPEAT_LIMIT}
    always_worn_out = worn_out_families | {_UCB_UNCERTAINTY_TAG}

    def tag_ok(key):
        return not (STRATEGY_HINT_TAGS.get(key, set()) & always_worn_out)

    def key_ok(key):
        return key not in hint_worn_out

    all_items = list(STRATEGY_HINTS.items())
    filter_stages = [
        lambda k: tag_ok(k) and key_ok(k),
        lambda k: key_ok(k),
        lambda k: tag_ok(k),
        lambda k: True,
    ]
    for stage in filter_stages:
        candidates = [(k, d) for k, d in all_items if stage(k)]
        if candidates:
            return candidates[int(rng.integers(len(candidates)))]
    return all_items[0]  # unreachable — the last stage always matches


# ── Exploitation lane: jittering the champion's own weight constants ───────
#
# Found on v4 run1's --resume continuation (generation 17-20): with parts
# 1-3 above all active and confirmed working (family_attempt_counts showed
# genuine rotation across all named families AND no generic_repeat trigger
# — recent_docstrings held 5 genuinely distinct ideas), the run was STILL
# stagnant. This isn't the mode-collapse failure the guards above exist
# for — the model IS diversifying. The cause is different: the incumbent
# champion (an acq_value_norm-based blend, already close to real EGBO's
# own performance per run_2b_diagnostic_v4.py's Step A) is hard to beat
# with a full from-scratch rewrite, and the one thing most likely to
# actually improve on it — small numeric refinement of ITS OWN weights
# (e.g. trying w_acq=0.85 instead of 0.9) — is exactly what
# champion_rehash_similarity bans once triggered, because a weight-tuned
# variant has the same high word-overlap as a lazy cosmetic reword. The
# guard can't tell "same idea, different prose" from "same structure,
# different constants" from docstring text alone.
#
# Rather than teaching the LLM to distinguish these (unreliable — see the
# hard-redirect comment above on why "please don't" text isn't a reliable
# enforcement mechanism), this adds a second, non-LLM child each stagnant
# generation: the champion's own code with its bare 0.NNN numeric literals
# (the pattern egbo_novelty_like's own w_acq=0.9/w_nov=0.1 follow, and
# most LLM-authored blend weights follow too) perturbed by a small random
# fraction. It bypasses the LLM, the annealing prompt, and every
# anti-repetition guard above entirely — this is deliberate local search
# around a formula that's already working, not a new proposal to classify
# or ban.
_WEIGHT_LITERAL_RE = re.compile(r"(?<![\w.])(0\.\d+)(?!\d)")
JITTER_FRACTION = 0.2

# Overfitting cap: jitter_champion_code hill-climbs against the SAME fixed
# set of training campaigns every generation — a smooth, low-dimensional
# search that can converge on a value fitting quirks of that fixed set
# rather than a genuinely better setting (the classic "tuned too many
# times against the same validation split" failure), more efficiently
# than free-form LLM proposals would. If a jitter-sourced child becomes
# the population's champion several generations running, that's exactly
# the regime where this risk is highest — nothing has re-challenged it
# with a structurally different, LLM-proposed idea in a while. Once a
# jitter-sourced champion has held the top spot for
# JITTER_CHAMPION_STREAK_LIMIT consecutive generations, the jitter lane is
# skipped for one generation, forcing that slot back to a real LLM
# proposal so at least one non-jitter idea competes against it. This
# doesn't fix overfitting on its own — validating the eventual champion
# against the held-out set (same standard used throughout this project,
# e.g. run_2b_diagnostic_v4.py) is still the real check — it just stops
# the loop from spending unlimited consecutive generations doing nothing
# but narrow scalar-tuning on a fixed training set.
JITTER_CHAMPION_STREAK_LIMIT = 3


def jitter_champion_code(champion_code: str, rng: np.random.Generator,
                          jitter_frac: float = JITTER_FRACTION) -> str:
    """Returns a copy of champion_code with every bare 0.NNN numeric
    literal perturbed by +/- jitter_frac (relative, with a small additive
    floor so near-zero constants can still move), clipped to [0.01, 0.99].
    Only matches literals of the exact form 0.NNN (not preceded by a digit
    or dot) — deliberately does NOT touch integers, array indices/loop
    bounds, or exponential-notation epsilons like 1e-8, so it can't
    corrupt code structure the way a blind numeric-literal regex would.
    If champion_code has no such literals, returns it unchanged (the
    caller still gets a valid, if identical, child rather than an error).
    """
    def _perturb(m):
        val = float(m.group(1))
        delta = rng.uniform(-jitter_frac, jitter_frac) * max(val, 0.05)
        new_val = min(0.99, max(0.01, val + delta))
        text = f"{new_val:.4f}".rstrip("0").rstrip(".")
        return text if "." in text else text + ".0"
    return _WEIGHT_LITERAL_RE.sub(_perturb, champion_code)


def _clean_llm_code(raw: str) -> str:
    """Strip a leading/trailing ```python fence off a raw LLM response.
    Module-level (not a local closure) so both llm_propose_child's normal
    path and attempt_free_invention's extra call below can share it."""
    raw = raw.strip()
    for fence in ["```python", "```"]:
        if raw.startswith(fence):
            raw = raw[len(fence):]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


HARD_REDIRECT_COUNT_NAME = "hard_redirect_count"
FREE_INVENTION_PERIOD = 3


def attempt_free_invention(best_doc: str, recent_docstrings: list,
                            family_attempt_counts: dict, system_prompt: str,
                            model: str, rng: np.random.Generator,
                            objective_names: list, domain_description: str,
                            feature_dim: int, temperature: float):
    """One extra LLM call, tried before the hard redirect's guaranteed-
    escape literal-catalog-implementation (see that block's own comment)
    every FREE_INVENTION_PERIOD-th time this generation's model has
    exhausted the soft annealing note. Returns cleaned code if the result
    passes the SAME champion/generic rehash checks used everywhere else,
    else None so the caller falls through to the forced-catalog redirect.

    Why this exists: the hard redirect forces literal implementation of
    one MECHANISM_FAMILIES/STRATEGY_HINTS entry — reliable (see that
    block's history for why prose alone wasn't), but it can never again
    produce something like v3 run1's `gen5_child0` (progress-weighted
    front-range-normalized GP mean/std blend) — a genuinely novel idea
    that came from ordinary, unconstrained proposal generation, not from
    any catalog. Once a run is deep enough into repeated hard-redirects
    (which is exactly when this function's caller runs), EVERY child is
    catalog-bound and that kind of discovery becomes structurally
    impossible for the rest of the run.

    Why not just try this every time instead of literal-catalog-forcing:
    that's what the ORIGINAL soft annealing note already does ("invent a
    different mechanism entirely" is already one of its stated options) —
    and it demonstrably doesn't reliably work once the model is fixated
    (see the hard-redirect block's own history: prose alone let it keep
    rewording the same idea for 14+ generations in run2). This function is
    the same free-form ask, but with an explicit, comprehensive banned-
    ideas list (the champion, recent circling docstrings, AND every
    STRATEGY_HINTS entry already tried via a redirect this streak) rather
    than a general "don't repeat yourself," and its output is checked
    before being trusted rather than assumed to have worked — if it's
    still a rehash, the guaranteed fallback still runs right after it.
    Only every 3rd attempt, not every time, so most stuck generations
    still get the reliable escape without paying for a probably-failing
    extra LLM call each time.
    """
    banned_bits = [f"the current best (\"{best_doc}\")"]
    for d in (recent_docstrings or [])[-4:]:
        if d and d != best_doc:
            banned_bits.append(f"\"{d}\"")
    tried_hint_keys = [k[len("hint_"):] for k, v in family_attempt_counts.items()
                        if k.startswith("hint_") and v > 0]
    for key in tried_hint_keys:
        desc = STRATEGY_HINTS.get(key)
        if desc:
            banned_bits.append(f"\"{desc}\"")
    banned_text = "; ".join(banned_bits)

    prompt = (
        f"Design a new modifier (correction term added to acq_value_norm "
        f"automatically — see the contract above) from scratch for a "
        f"{domain_description} with objectives {objective_names} "
        f"(feature_dim={feature_dim}).\n\n"
        f"Every recent proposal this stagnation streak has just been a "
        f"reworded version of one of the following already-tried ideas — "
        f"do NOT propose anything that resembles ANY of them, not even "
        f"with different vocabulary or a minor twist on the same "
        f"underlying formula: {banned_text}.\n\n"
        f"Invent a genuinely different mechanism, built on different "
        f"underlying math from every idea listed above. Remember the "
        f"required one-line docstring as the first statement."
    )
    import ollama  # local import, matches llm_propose_child's own — see
    # its comment: keeps the mock-mode code path (evolve_af_v4.py --mock)
    # free of an ollama dependency it never uses. This function is a
    # SEPARATE top-level function from llm_propose_child, not nested
    # inside it, so it does NOT inherit llm_propose_child's own local
    # `import ollama` — that was missing here entirely until found via
    # v4 run1's live behaviour post-resume (gens 111-114: 8/9 calls
    # fell back to mock crossover). Every call into this function raised
    # NameError immediately, which propagated uncaught out of
    # llm_propose_child to make_child's except block — not just failing
    # the free-invention attempt, but the entire child generation. Worse,
    # because the exception fired before HARD_REDIRECT_COUNT_NAME ever
    # got incremented (see the call site in llm_propose_child), hr_count
    # stayed stuck at 0 forever, so `hr_count % FREE_INVENTION_PERIOD ==
    # 0` was true on EVERY call — once a run reaches the hard-redirect
    # state (which run1 already had, durably, from before this bug was
    # introduced), essentially every subsequent generation failed. A unit
    # test against this function passed anyway because it monkeypatched
    # `evolve_af_v4.ollama` directly onto the module, which silently
    # created the very module-level global whose absence was the bug —
    # never exercising the real unpatched import path.
    resp = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt}],
        options={"temperature": min(1.2, temperature + 0.15), "num_predict": 1024,
                 "repeat_penalty": 1.3, "num_ctx": 8192,
                 "seed": int(rng.integers(1_000_000))},
        keep_alive="30m",
    )
    code = _clean_llm_code(resp["message"]["content"])
    doc = extract_af_docstring(code, af_function_name=AF_FUNCTION_NAME_V6)
    champion_sim = champion_rehash_similarity(doc, best_doc)
    generic_sim = generic_repeat_similarity(doc, recent_docstrings)
    if champion_sim >= CHAMPION_REHASH_THRESHOLD or generic_sim >= CHAMPION_REHASH_THRESHOLD:
        return None
    return code


def llm_propose_child(parent_a: dict, parent_b: dict, best_so_far: dict, steps: list,
                       model: str, rng: np.random.Generator,
                       objective_names: list, domain_description: str,
                       feature_dim: int = 16, stagnant_generations: int = 0,
                       family_attempt_counts: dict = None,
                       recent_docstrings: list = None) -> str:
    """Real-LLM crossover/mutation — same shape as evolve_af.py's
    llm_propose_child, but mean_margin/LOC in the prompt now refer to
    campaign-level, margin-based fitness (see evaluate_af_2b's FITNESS
    METRIC note), and the system prompt is domain-generic.

    stagnant_generations: how many generations since the population's best
    fitness last improved (tracked by run_evolution's loop, reset to 0 on
    any improvement). Used to ANNEAL this call rather than stopping the
    run early: as this grows, temperature rises (more willing to deviate
    from the parents) and, past a threshold, the prompt explicitly names
    the stagnation and asks for a structurally different mechanism instead
    of a small tweak. This targets the actual observed failure mode
    directly — the coatings run's gen 6-20 plateau was 14 generations of
    near-identical small variations on the current best, not an exhausted
    search space (see af_interface_v2.py's 8-hint set, most of which never
    survived past gen 0) — rather than a stagnation-based EARLY STOP, which
    would have just given up before finding out whether a bigger nudge
    could have escaped the plateau.

    family_attempt_counts: {family_name: count}, how many stagnation-
    triggered children THIS stagnation streak have already been classified
    into each MECHANISM_FAMILIES entry (run_evolution resets this to {}
    every time the population's best fitness improves — see
    _run_one_generation). Fixes a mode-collapse failure found on the
    tunable-domain run1: without this, the annealing note below just lists
    ALL example mechanisms every time, and the model anchors on whichever
    one it tried first, re-proposing near-verbatim restatements of it for
    dozens of generations (see MECHANISM_FAMILIES' docstring above) instead
    of actually rotating through the list as intended.

    recent_docstrings: rolling window of recently-proposed children's
    docstrings this stagnation streak (see GENERIC_REPEAT_NAME above) —
    used to detect/ban fixation on an idea that isn't the champion and
    isn't one of the named MECHANISM_FAMILIES either.
    """
    import ollama
    system_prompt = _build_system_prompt(objective_names, domain_description, feature_dim)
    diag = _campaign_diagnostic_text(steps)
    best_doc = best_so_far.get("docstring") or "(no docstring recorded)"
    prompt = (
        f"Best-so-far in the population (\"{best_so_far['id']}\", "
        f"avg_z={best_so_far['mean_margin']:+.3f}): \"{best_doc}\"\n\n"
        f"Parent A (avg_z={parent_a['mean_margin']:+.3f}, LOC={parent_a['loc']}):\n"
        f"```python\n{parent_a['code']}\n```\n\n"
        f"Parent B (avg_z={parent_b['mean_margin']:+.3f}, LOC={parent_b['loc']}):\n"
        f"```python\n{parent_b['code']}\n```\n\n"
        f"Representative training campaign summary:\n{diag}\n\n"
        f"Write a new modifier that combines or improves on these two parents' "
        f"corrections (remember: this is a correction term added to "
        f"acq_value_norm automatically, not a full score — see the contract "
        f"above). Remember the required one-line docstring as the first statement."
    )

    # Annealing: only kicks in once stagnation is established (threshold 2
    # — gen 0-1 improving normally shouldn't trigger this), scales with
    # how long it's persisted, capped so it never becomes pure noise.
    STAGNATION_THRESHOLD = 2
    temperature = min(1.2, 0.7 + 0.06 * max(0, stagnant_generations - STAGNATION_THRESHOLD))
    if stagnant_generations >= STAGNATION_THRESHOLD:
        family_attempt_counts = family_attempt_counts or {}
        worn_out = [name for name, _desc, _kw in MECHANISM_FAMILIES
                    if family_attempt_counts.get(name, 0) >= FAMILY_REPEAT_LIMIT]
        fresh = [(name, desc) for name, desc, _kw in MECHANISM_FAMILIES
                 if name not in worn_out]
        # Sort remaining families least-tried-first so the note always
        # leads with whatever the model hasn't converged on yet.
        fresh.sort(key=lambda nd: family_attempt_counts.get(nd[0], 0))

        note = (
            f"\n\nNOTE: the population's best fitness has not improved for "
            f"{stagnant_generations} generations — small variations on the "
            f"current best (different fixed/decaying uncertainty weights, "
            f"minor novelty tweaks) have stopped working. Do NOT propose "
            f"another small weight variation this time. Instead, try a "
            f"STRUCTURALLY different mechanism from what's shown above."
        )
        if worn_out:
            worn_out_descs = "; ".join(
                desc for name, desc, _kw in MECHANISM_FAMILIES if name in worn_out)
            note += (
                f" Do NOT propose another variant of: {worn_out_descs} — "
                f"that idea has already been retried {FAMILY_REPEAT_LIMIT}+ "
                f"times this stagnation streak without improving fitness, "
                f"so another restatement of it is very unlikely to help."
            )
        # Champion-rehash guard (added after run2 — see CHAMPION_REHASH_NAME's
        # docstring): fires independently of the worn_out/fresh family list
        # above, because rewording the incumbent's own idea isn't one of the
        # suggested families, it's whatever the current best_so_far happens
        # to be — a moving target the keyword classifier above can't see.
        if family_attempt_counts.get(CHAMPION_REHASH_NAME, 0) >= FAMILY_REPEAT_LIMIT:
            note += (
                f" Also: do NOT just reword the current best's own idea "
                f"(\"{best_doc}\") in different vocabulary — {family_attempt_counts[CHAMPION_REHASH_NAME]} "
                f"of your recent proposals this streak were already just "
                f"restatements of that same mechanism (same core "
                f"formula/terms, different wording) and none of them "
                f"improved fitness either. A new mechanism means different "
                f"underlying math, not a renamed version of the current "
                f"best's math."
            )
        # Generic-repeat guard (added after v4 run1 — see GENERIC_REPEAT_NAME's
        # docstring): catches fixation on an idea that is neither the
        # champion nor one of the four named families (e.g. repeatedly
        # rewording "UCB score + inverse-distance novelty" — a pattern the
        # keyword/champion checks above both missed in that run).
        if family_attempt_counts.get(GENERIC_REPEAT_NAME, 0) >= FAMILY_REPEAT_LIMIT:
            recent_examples = "; ".join(
                f"\"{d}\"" for d in (recent_docstrings or [])[-2:])
            note += (
                f" Also: {family_attempt_counts[GENERIC_REPEAT_NAME]} of your "
                f"recent proposals this streak were near-duplicates of EACH "
                f"OTHER (not the champion, not a named family above — some "
                f"other idea you kept rewording), e.g. {recent_examples}. "
                f"Do NOT propose another variant of that idea either — pick "
                f"a mechanism you have not already tried this streak."
            )
        if fresh:
            fresh_descs = "; or ".join(desc for _name, desc in fresh)
            note += (
                f" Try one of these instead (pick ONE, implement it as the "
                f"dominant idea, not a minor addition to the "
                f"mu_sum+uncertainty pattern above): {fresh_descs}."
            )
        else:
            note += (
                " Every listed mechanism above has already been tried "
                "repeatedly this streak without improving fitness — either "
                "compose two of them together in a genuinely new way, or "
                "invent a different mechanism entirely that doesn't match "
                "any of them."
            )
        prompt += note

    resp = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_predict": 1024,
                 "repeat_penalty": 1.3,
                 "num_ctx": 8192,
                 "seed": int(rng.integers(1_000_000))},
        # See _generate's comment above: avoid eviction/reload during the
        # long GP-fit/optimize_acqf gap between LLM calls.
        keep_alive="30m",
    )

    code = _clean_llm_code(resp["message"]["content"])

    # Hard redirect — found necessary on run2's --resume continuation: the
    # champion-rehash TEXTUAL ban above (added after run2's first 50
    # generations) fires correctly (family_attempt_counts["champion_rehash"]
    # climbed 0->11 over the next 14 generations) but the model kept
    # producing near-identical rehashes ANYWAY (rehash_sim 0.35-0.73)
    # even several calls after the ban text was already in its prompt —
    # telling this model "don't do X" in prose is not a reliable
    # enforcement mechanism once it's fixated on an idea, likely because
    # the anchor (best_so_far's own docstring/code, quoted verbatim in
    # every prompt) is stronger than the negative instruction telling it
    # not to use it. Rather than adding a THIRD round of "please don't"
    # text, this checks the actual output post-hoc and, if it's still a
    # rehash after the ban was already active, throws it away and
    # regenerates ONCE from a prompt that structurally removes the anchor
    # (no parent code, no best-so-far reference) and hands the model one
    # concrete, least-tried idea to implement instead of leaving the
    # choice open to drift back to what it just read.
    if stagnant_generations >= STAGNATION_THRESHOLD:
        family_attempt_counts = family_attempt_counts or {}
        recent_docstrings = recent_docstrings or []
        champion_banned = (
            family_attempt_counts.get(CHAMPION_REHASH_NAME, 0) >= FAMILY_REPEAT_LIMIT)
        generic_banned = (
            family_attempt_counts.get(GENERIC_REPEAT_NAME, 0) >= FAMILY_REPEAT_LIMIT)
        already_banned = champion_banned or generic_banned
        if already_banned:
            doc = extract_af_docstring(code, af_function_name=AF_FUNCTION_NAME_V6)
            champion_sim = champion_rehash_similarity(doc, best_doc) if champion_banned else 0.0
            generic_sim = (generic_repeat_similarity(doc, recent_docstrings)
                           if generic_banned else 0.0)
            if champion_sim >= CHAMPION_REHASH_THRESHOLD or generic_sim >= CHAMPION_REHASH_THRESHOLD:
                # Free-invention attempt, tried first every FREE_INVENTION_PERIOD-th
                # time this branch fires this streak (see HARD_REDIRECT_COUNT_NAME's
                # docstring below for why this exists and isn't the default).
                hr_count = family_attempt_counts.get(HARD_REDIRECT_COUNT_NAME, 0)
                if hr_count % FREE_INVENTION_PERIOD == 0:
                    invented_code = attempt_free_invention(
                        best_doc, recent_docstrings, family_attempt_counts,
                        system_prompt, model, rng, objective_names,
                        domain_description, feature_dim, temperature)
                    if invented_code is not None:
                        family_attempt_counts[HARD_REDIRECT_COUNT_NAME] = hr_count + 1
                        return invented_code
                family_attempt_counts[HARD_REDIRECT_COUNT_NAME] = hr_count + 1

                worn_out_now = {name for name, _desc, _kw in MECHANISM_FAMILIES
                                 if family_attempt_counts.get(name, 0) >= FAMILY_REPEAT_LIMIT}
                fresh_now = [desc for name, desc, _kw in MECHANISM_FAMILIES
                             if name not in worn_out_now]
                if fresh_now:
                    redirect_desc = fresh_now[int(rng.integers(len(fresh_now)))]
                else:
                    redirect_key, redirect_desc = unexhausted_strategy_hint(
                        rng, family_attempt_counts, worn_out_now, best_doc)
                    # Tally the exact hint key immediately — this mutates
                    # the SAME dict object _run_one_generation holds (see
                    # unexhausted_strategy_hint's docstring), not a copy:
                    # family_attempt_counts is only ever reassigned to a
                    # NEW dict via `x or {}` when it's empty/falsy, and it
                    # can't be empty here (champion_banned/generic_banned
                    # both require an entry already >= FAMILY_REPEAT_LIMIT).
                    family_attempt_counts[f"hint_{redirect_key}"] = (
                        family_attempt_counts.get(f"hint_{redirect_key}", 0) + 1)
                if champion_sim >= CHAMPION_REHASH_THRESHOLD:
                    reworded_thing = f"the current best (\"{best_doc}\")"
                    n_repeats = family_attempt_counts.get(CHAMPION_REHASH_NAME, 0)
                else:
                    reworded_thing = "an idea you already tried repeatedly this streak"
                    n_repeats = family_attempt_counts.get(GENERIC_REPEAT_NAME, 0)
                redirect_prompt = (
                    f"Implement the following acquisition strategy as "
                    f"a modifier (correction term added to acq_value_norm "
                    f"automatically), for a {domain_description} with "
                    f"objectives {objective_names} (feature_dim={feature_dim}):"
                    f"\n\n\"{redirect_desc}\"\n\n"
                    f"This is a hard requirement, not a suggestion: your "
                    f"last {n_repeats} proposals this stagnation streak were "
                    f"all just reworded versions of {reworded_thing} — "
                    f"different vocabulary, same underlying formula — and "
                    f"none of them improved fitness. Do NOT reference, "
                    f"resemble, or partially reuse that formula this time. "
                    f"Implement ONLY the strategy described above, built "
                    f"from scratch."
                )
                resp2 = ollama.chat(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": redirect_prompt}],
                    options={"temperature": min(1.2, temperature + 0.1),
                             "num_predict": 1024, "repeat_penalty": 1.3,
                             "num_ctx": 8192,
                             "seed": int(rng.integers(1_000_000))},
                    keep_alive="30m",
                )
                code = _clean_llm_code(resp2["message"]["content"])

    return code


# ── Evolution loop ──────────────────────────────────────────────────────────

def tournament_select(population: list, rng: np.random.Generator, k: int = 3) -> dict:
    idx = rng.choice(len(population), size=min(k, len(population)), replace=False)
    contenders = [population[i] for i in idx]
    return max(contenders, key=lambda p: p["fitness"])


def make_child(parent_a: dict, parent_b: dict, best_so_far: dict, steps: list,
               model: str, rng: np.random.Generator,
               objective_names: list, domain_description: str, feature_dim: int = 16,
               stagnant_generations: int = 0, family_attempt_counts: dict = None,
               recent_docstrings: list = None):
    """Identical mechanism to evolve_af.py's make_child, plus annealing
    under stagnation — see llm_propose_child's docstring. v6 is real-LLM-
    only (no mock-mode crossover system — see af_interface_v6.py's module
    docstring for why: alpha/mechanism choice for a delta-seed modifier
    needs real reasoning, which mock mode's fixed TERM_TEMPLATES
    vocabulary — designed for full score_pool rewrites — has no analog
    for). On an LLM failure, falls back to a JITTERED COPY of best_so_far's
    own code (jitter_champion_code, defined above this function — a small
    numeric-literal perturbation, contract-agnostic and guaranteed
    sandbox-valid since best_so_far already passed it once) rather than
    v5's mock crossover fallback."""
    try:
        code = llm_propose_child(parent_a, parent_b, best_so_far, steps, model, rng,
                                  objective_names, domain_description, feature_dim,
                                  stagnant_generations=stagnant_generations,
                                  family_attempt_counts=family_attempt_counts,
                                  recent_docstrings=recent_docstrings)
        return code, True
    except Exception as e:
        warnings.warn(
            f"llm_propose_child failed ({type(e).__name__}: {e}) — "
            f"falling back to a jittered copy of best_so_far for this child.\n"
            + traceback.format_exc())
        return jitter_champion_code(best_so_far["code"], rng), False


def save_checkpoint(checkpoint_path: pathlib.Path, generation: int, population: list,
                     history: list, rng: np.random.Generator, stagnant_generations: int,
                     best_fitness_ever: float, n_llm_calls: int, n_llm_failures: int,
                     family_attempt_counts: dict = None,
                     recent_docstrings: list = None,
                     jitter_champion_streak: int = 0) -> None:
    """
    Writes ONE checkpoint file (overwritten every generation, not
    accumulated) capturing everything --resume needs to continue this run
    from exactly where it left off: the full population (plain JSON-
    serializable dicts — no term_weights field in v6, unlike v5; see
    af_interface_v6.py's module docstring for why there's no mock-mode
    system to serialize), history, the RNG's own bit-generator state (so
    --resume doesn't just reseed from scratch and silently replay the
    same tournament/mutation draws already used), and
    the pieces of loop state (stagnant_generations, best_fitness_ever,
    family_attempt_counts) that live outside `history`/`population` and
    would otherwise reset to their gen-0 defaults on resume, silently
    breaking annealing (and, for family_attempt_counts, re-enabling the
    exact mode-collapse repetition it exists to stop — see
    MECHANISM_FAMILIES' docstring).
    """
    payload = {
        "generation": generation,
        "population": population,
        "history": history,
        "rng_state": rng.bit_generator.state,
        "stagnant_generations": stagnant_generations,
        "best_fitness_ever": best_fitness_ever,
        "n_llm_calls": n_llm_calls,
        "n_llm_failures": n_llm_failures,
        "family_attempt_counts": family_attempt_counts or {},
        "recent_docstrings": recent_docstrings or [],
        "jitter_champion_streak": jitter_champion_streak,
    }
    tmp = checkpoint_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(checkpoint_path)  # atomic on POSIX — a crash mid-write never
                                    # corrupts the last good checkpoint


def load_checkpoint(checkpoint_path: pathlib.Path) -> dict:
    with open(checkpoint_path) as f:
        return json.load(f)


_MULTI_DOMAIN_DESCRIPTION = (
    "a multi-objective Bayesian optimization campaign, trained across "
    "SEVERAL experimental domains simultaneously (different objective "
    "counts and feature dimensionality per domain) — do not assume a "
    "fixed number of objectives, a specific feature dimensionality, or "
    "any particular objective names; read context['objective_names'] and "
    "context['pool'][i]['x']'s own length dynamically instead of "
    "hardcoding any of these"
)


def run_evolution(domain_configs: dict, pop_size: int,
                   n_generations: int, n_offspring: int, gamma: float,
                   model: str, seed: int,
                   objective_names: list = None, domain_description: str = None,
                   feature_dim: int = None, directions: list = None,
                   log_dir: pathlib.Path = None,
                   gate_z_threshold: float = None,
                   checkpoint_path: pathlib.Path = None,
                   resume_state: dict = None) -> dict:
    """
    domain_configs: ORDERED {domain_name: {"training_logs", "baseline_hvs",
    "oracle_family", "n_fitness_seeds"}} — order IS evaluate_af_multi_domain's
    gate sequence (tunable, coatings, mAb — cheapest/cleanest first, most
    expensive/noisiest last; see that function's own docstring and the
    af-evolution branch design conversation for why).

    checkpoint_path: if given, a checkpoint is written (overwriting any
    previous one at this path) after gen 0's initial population AND after
    every subsequent generation — so stopping the run at any point (Ctrl-C,
    a crash, or just letting --n_generations finish) always leaves a
    resumable, complete-generation checkpoint, never a half-written one.

    resume_state: if given (the dict load_checkpoint returns), skips
    building a fresh population entirely and continues from this state's
    generation/population/history/rng/stagnant_generations/
    best_fitness_ever. n_generations is interpreted as the TOTAL target
    generation count in this case (matching the same flag's meaning on a
    fresh run) — e.g. run once with --n_generations 3, inspect the
    checkpoint, then rerun with --resume <path> --n_generations 10 to
    continue on to generation 10, not "10 more" on top of the 3 already
    done. If the resumed generation is already >= n_generations, the run
    exits immediately after re-reporting the loaded state — a no-op, not
    an error, so re-running the same command twice is harmless.
    """
    # No single-domain default here (unlike v4's ORACLE_DEFAULTS["excipient"]
    # fallback) — a multi-domain run has no one "the domain" to default to;
    # objective_names/domain_description/feature_dim must either be passed
    # explicitly or fall back to the deliberately generic, domain-agnostic
    # text below, which tells the model NOT to assume a fixed shape rather
    # than silently anchoring it on whichever domain happened to be first.
    objective_names = objective_names or ["obj_1", "obj_2", "..."]
    domain_description = domain_description or _MULTI_DOMAIN_DESCRIPTION
    # feature_dim stays None if not given — interpolated into prompt text
    # as "d" (see _build_system_prompt's feature_dim_text), an honest
    # signal there's no single fixed value across domains, rather than a
    # specific-but-wrong number.
    first_domain_cfg = next(iter(domain_configs.values()))
    # Bug found via v5's own first smoke test: `directions` is fundamentally
    # a PER-DOMAIN property (tunable is 2-objective min/min, mAb is
    # 3-objective max/max/min, etc.) — the old single top-level `directions`
    # param defaulted to None here, which fell through to to_allmax's own
    # default (excipient's 3-objective convention) regardless of which
    # domain was actually first, and crashed with an IndexError the moment
    # tunable (2 objectives) was gate-sequence-first. Read it from the
    # first domain's OWN config instead — main() populates domain_configs
    # entries with "directions" precisely so this works regardless of gate
    # order. Falls back to the (now-unused-by-main but still overridable)
    # `directions` parameter only if a domain_config entry doesn't carry
    # its own (e.g. a caller building domain_configs by hand).
    first_domain_directions = first_domain_cfg.get("directions", directions)
    pseudo_steps = build_pseudo_steps(first_domain_cfg["training_logs"],
                                       first_domain_cfg["baseline_hvs"],
                                       directions=first_domain_directions)

    if resume_state is not None:
        rng = np.random.default_rng()
        rng.bit_generator.state = resume_state["rng_state"]
        population = resume_state["population"]
        history = resume_state["history"]
        stagnant_generations = resume_state["stagnant_generations"]
        best_fitness_ever = resume_state["best_fitness_ever"]
        n_llm_calls = resume_state["n_llm_calls"]
        n_llm_failures = resume_state["n_llm_failures"]
        # .get(..., {}) for backward compatibility with checkpoints written
        # before family_attempt_counts existed.
        family_attempt_counts = resume_state.get("family_attempt_counts", {})
        # .get(..., []) for backward compatibility with checkpoints written
        # before recent_docstrings existed (see GENERIC_REPEAT_NAME above).
        recent_docstrings = resume_state.get("recent_docstrings", [])
        # .get(..., 0) for backward compatibility with checkpoints written
        # before jitter_champion_streak/jitter_champion_code existed.
        jitter_champion_streak = resume_state.get("jitter_champion_streak", 0)
        start_gen = resume_state["generation"] + 1
        print(f"Resumed from checkpoint at generation {resume_state['generation']} "
              f"(best fitness={population[0]['fitness']:.4f}, id={population[0]['id']!r}). "
              f"Continuing to generation {n_generations}.")
        if start_gen > n_generations:
            print(f"Resumed generation ({resume_state['generation']}) already >= "
                  f"--n_generations ({n_generations}) — nothing to do.")
            return {"population": population, "history": history,
                    "n_llm_calls": n_llm_calls, "n_llm_failures": n_llm_failures}
        for gen in range(start_gen, n_generations + 1):
            population, history, stagnant_generations, best_fitness_ever, n_llm_calls, \
                n_llm_failures, family_attempt_counts, recent_docstrings, \
                jitter_champion_streak = _run_one_generation(
                    gen, population, history, stagnant_generations, best_fitness_ever,
                    n_llm_calls, n_llm_failures, domain_configs, gamma,
                    pop_size, n_offspring, model, rng, objective_names,
                    domain_description, feature_dim, log_dir,
                    gate_z_threshold, pseudo_steps, family_attempt_counts, recent_docstrings,
                    jitter_champion_streak)
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, gen, population, history, rng,
                                 stagnant_generations, best_fitness_ever,
                                 n_llm_calls, n_llm_failures, family_attempt_counts,
                                 recent_docstrings, jitter_champion_streak)
        _print_llm_summary(n_llm_calls, n_llm_failures)
        return {"population": population, "history": history,
                "n_llm_calls": n_llm_calls, "n_llm_failures": n_llm_failures}

    rng = np.random.default_rng(seed)

    population = []
    for name, code in SEED_PROGRAMS.items():
        result = evaluate_af_multi_domain(code, domain_configs, gamma, log_dir=log_dir,
                                           gate_z_threshold=gate_z_threshold)
        population.append({"id": name, "code": code, **result})

    # Generation-0 bootstrap: ask the LLM to write modifier(context) for
    # each one-line STRATEGY_HINT, rather than shipping a hand-written
    # implementation for every strategy (see af_interface_v2.py's module
    # docstring — same rationale v2-v5 used, just for the delta-seed
    # contract now). v6 has no mock mode, so a hint-generation failure
    # falls back to a JITTERED COPY of the one hand-written seed
    # (uncertainty_modifier) rather than a random mock program — always
    # sandbox-valid, distinct enough per-hint via a fresh rng draw each
    # time.
    _fallback_seed_code = next(iter(SEED_PROGRAMS.values()))
    for hint_name, hint_text in STRATEGY_HINTS.items():
        try:
            code = llm_generate_seed_from_hint(hint_text, model, rng,
                                                objective_names, domain_description,
                                                feature_dim)
        except Exception as e:
            warnings.warn(
                f"llm_generate_seed_from_hint failed for {hint_name!r} "
                f"({type(e).__name__}: {e}) — falling back to a jittered "
                f"copy of the seed program for this hint slot.\n" + traceback.format_exc())
            code = jitter_champion_code(_fallback_seed_code, rng)
        result = evaluate_af_multi_domain(code, domain_configs, gamma, log_dir=log_dir,
                                           gate_z_threshold=gate_z_threshold)
        population.append({"id": f"hint_{hint_name}", "code": code, **result})

    # Pad with jittered copies of already-built population members for
    # initial diversity beyond the seed/hint-generated entries — v5's
    # random-term-weight padding has no v6 analog (no mock term-weight
    # system), so this reuses jitter_champion_code (already a first-class
    # tool in this file for the stagnation exploitation lane) against a
    # randomly-drawn EXISTING member instead.
    for i in range(max(0, pop_size - len(population))):
        base = population[rng.integers(len(population))]
        code = jitter_champion_code(base["code"], rng)
        result = evaluate_af_multi_domain(code, domain_configs, gamma, log_dir=log_dir,
                                           gate_z_threshold=gate_z_threshold)
        population.append({"id": f"jitter_init_{i}", "code": code, **result})
    population = sorted(population, key=lambda p: -p["fitness"])[:max(pop_size, len(population))]

    history = [{"generation": 0, "best_fitness": population[0]["fitness"],
                "best_mean_margin": population[0]["mean_margin"],
                "best_win_rate": population[0]["win_rate"],
                "best_domain_scores": population[0]["domain_scores"],
                "best_domain_zs": population[0]["domain_zs"]}]
    print(f"gen 0: best fitness={population[0]['fitness']:.4f} "
          f"avg_z={population[0]['mean_margin']:+.4f} "
          f"domains_passed={population[0]['win_rate']:.2f} "
          f"domain_scores={population[0]['domain_scores']} "
          f"domain_zs={population[0]['domain_zs']} "
          f"({population[0]['id']})")

    n_llm_calls, n_llm_failures = 0, 0
    best_fitness_ever = population[0]["fitness"]
    stagnant_generations = 0
    family_attempt_counts = {}
    recent_docstrings = []
    jitter_champion_streak = 0

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, 0, population, history, rng,
                         stagnant_generations, best_fitness_ever, n_llm_calls, n_llm_failures,
                         family_attempt_counts, recent_docstrings, jitter_champion_streak)

    for gen in range(1, n_generations + 1):
        population, history, stagnant_generations, best_fitness_ever, n_llm_calls, \
            n_llm_failures, family_attempt_counts, recent_docstrings, \
            jitter_champion_streak = _run_one_generation(
                gen, population, history, stagnant_generations, best_fitness_ever,
                n_llm_calls, n_llm_failures, domain_configs, gamma,
                pop_size, n_offspring, model, rng, objective_names,
                domain_description, feature_dim, log_dir,
                gate_z_threshold, pseudo_steps, family_attempt_counts, recent_docstrings,
                jitter_champion_streak)
        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, gen, population, history, rng,
                             stagnant_generations, best_fitness_ever,
                             n_llm_calls, n_llm_failures, family_attempt_counts,
                             recent_docstrings, jitter_champion_streak)

    _print_llm_summary(n_llm_calls, n_llm_failures)
    return {"population": population, "history": history,
            "n_llm_calls": n_llm_calls, "n_llm_failures": n_llm_failures}


def _run_one_generation(gen, population, history, stagnant_generations, best_fitness_ever,
                         n_llm_calls, n_llm_failures, domain_configs, gamma,
                         pop_size, n_offspring, model, rng, objective_names,
                         domain_description, feature_dim, log_dir,
                         gate_z_threshold, pseudo_steps, family_attempt_counts=None,
                         recent_docstrings=None, jitter_champion_streak=0):
    """One generation's worth of run_evolution's loop body, factored out so
    both the fresh-run path and the --resume path (which needs to run an
    arbitrary sub-range of generations, not always starting at 1) share the
    exact same logic rather than two copies that could drift apart.

    family_attempt_counts: see llm_propose_child's docstring and
    MECHANISM_FAMILIES above. Passed in read-only for this generation's
    prompt-building, then updated below (by classifying each stagnation-
    triggered child's docstring) and returned for the caller to persist.

    recent_docstrings: see GENERIC_REPEAT_NAME above. Same lifecycle as
    family_attempt_counts (passed in read-only, updated below, reset
    together on genuine improvement).

    jitter_champion_streak: see JITTER_CHAMPION_STREAK_LIMIT above — how
    many CONSECUTIVE generations a jitter-sourced child has held the
    population's #1 spot. Read-only here for this generation's use_jitter
    decision, updated below after the population resorts, returned for the
    caller to persist. NOT reset on improvement the way the other two
    counters are — a jitter-sourced champion improving fitness is exactly
    the case this cap exists to eventually interrupt, not a fresh streak
    to forgive.
    """
    family_attempt_counts = dict(family_attempt_counts or {})
    recent_docstrings = list(recent_docstrings or [])
    best_so_far = max(population, key=lambda p: p["fitness"])
    # LLM-facing reference pool: excludes jitter-sourced individuals. A
    # jitter child's docstring is byte-identical to the design it was
    # tuned from (jitter_champion_code only touches numeric literals), so
    # showing it to the LLM as "best-so-far" changes nothing textually —
    # but its FITNESS does get shown, and letting that fitness (inflated
    # by weight-tuning against this exact fixed training set) become the
    # bar every fresh idea has to clear is a real problem: it's a much
    # harder, more overfit-prone target than the honest LLM-designed
    # champion underneath it, and it can also get drawn as a tournament
    # parent, seeding crossover material with the tuned constant. Parent
    # selection and the "beat this" reference shown to the LLM both draw
    # from this filtered pool instead, so the LLM is always asked to
    # improve on genuine LLM designs — the jitter lane and elitism
    # (population[0], best_fitness_ever, checkpoint output) are untouched
    # and still see the true best, jitter included.
    llm_pool = [p for p in population if "_jitter" not in p["id"]] or population
    llm_reference = max(llm_pool, key=lambda p: p["fitness"])
    children = []
    for i in range(n_offspring):
        # Exploitation lane (see jitter_champion_code's docstring above):
        # one slot per stagnant generation goes to a non-LLM, weight-jittered
        # copy of the champion instead of a fresh proposal — deliberate
        # local search that the anti-repetition guards above would
        # otherwise ban. Only the LAST slot, and only once stagnation is
        # established, so early generations (already healthy per run1's
        # calls 0-15) and single-offspring runs (n_offspring=1) are
        # unaffected. Also skipped once a jitter-sourced child has held
        # the champion spot JITTER_CHAMPION_STREAK_LIMIT generations
        # running (see its docstring) — forces this slot back to a real
        # LLM proposal for one generation instead of tuning the same
        # fixed-training-set-fit scalar further.
        use_jitter = (stagnant_generations >= 2
                      and n_offspring >= 2 and i == n_offspring - 1
                      and jitter_champion_streak < JITTER_CHAMPION_STREAK_LIMIT)
        if use_jitter:
            code = jitter_champion_code(best_so_far["code"], rng)
            used_llm = False
        else:
            parent_a = tournament_select(llm_pool, rng)
            parent_b = tournament_select(llm_pool, rng)
            code, used_llm = make_child(parent_a, parent_b, llm_reference,
                                         pseudo_steps, model, rng,
                                         objective_names, domain_description,
                                         feature_dim,
                                         stagnant_generations=stagnant_generations,
                                         family_attempt_counts=family_attempt_counts,
                                         recent_docstrings=recent_docstrings)
            n_llm_calls += 1
            n_llm_failures += (not used_llm)
        result = evaluate_af_multi_domain(code, domain_configs, gamma, log_dir=log_dir,
                                           gate_z_threshold=gate_z_threshold)
        child_id = f"gen{gen}_jitter{i}" if use_jitter else f"gen{gen}_child{i}"
        children.append({"id": child_id, "code": code,
                          "used_llm": used_llm, **result})
        # Only tally this child against the anti-repetition counters if it
        # was actually generated under the annealing note (mirrors
        # llm_propose_child's STAGNATION_THRESHOLD) — pre-stagnation
        # diversity is already healthy (see run1's calls 0-15) and doesn't
        # need this pressure.
        if used_llm and stagnant_generations >= 2:
            child_doc = result.get("docstring")
            for family_name in classify_mechanism_families(child_doc):
                family_attempt_counts[family_name] = family_attempt_counts.get(family_name, 0) + 1
            # Champion-rehash tally (see CHAMPION_REHASH_NAME's docstring) —
            # compared against THIS generation's llm_reference, the same
            # docstring the child was actually prompted against (not
            # best_so_far, which may be a jitter child the LLM never saw).
            rehash_sim = champion_rehash_similarity(child_doc, llm_reference.get("docstring"))
            if rehash_sim >= CHAMPION_REHASH_THRESHOLD:
                family_attempt_counts[CHAMPION_REHASH_NAME] = (
                    family_attempt_counts.get(CHAMPION_REHASH_NAME, 0) + 1)
            # Generic-repeat tally (see GENERIC_REPEAT_NAME's docstring) —
            # compared against recent_docstrings BEFORE this child is added
            # to it, so a child can't trivially match itself.
            generic_sim = generic_repeat_similarity(child_doc, recent_docstrings)
            if generic_sim >= CHAMPION_REHASH_THRESHOLD:
                family_attempt_counts[GENERIC_REPEAT_NAME] = (
                    family_attempt_counts.get(GENERIC_REPEAT_NAME, 0) + 1)
            if child_doc:
                recent_docstrings.append(child_doc)
                recent_docstrings = recent_docstrings[-RECENT_DOCSTRING_WINDOW:]

    existing_sigs = {p["selection_signature"] for p in population}
    novel_children, n_duplicate = [], 0
    for c in children:
        if c["selection_signature"] in existing_sigs:
            n_duplicate += 1
            continue
        existing_sigs.add(c["selection_signature"])
        novel_children.append(c)

    population = sorted(population + novel_children, key=lambda p: -p["fitness"])[:pop_size]

    # jitter_champion_streak update (see JITTER_CHAMPION_STREAK_LIMIT above)
    # — tracks the #1 population slot specifically, not whether a jitter
    # child merely survived into the population. "_jitter" only ever
    # appears in ids this loop assigns (gen{N}_jitter{i}), never in a
    # seed/hint/mock/LLM child's id, so substring match is unambiguous.
    if "_jitter" in population[0]["id"]:
        jitter_champion_streak += 1
    else:
        jitter_champion_streak = 0

    # Stagnation tracking for next generation's annealing (see
    # llm_propose_child's docstring) — small epsilon tolerance so
    # floating-point-noise-level "improvement" doesn't reset the
    # counter and mask genuine stagnation.
    if population[0]["fitness"] > best_fitness_ever + 1e-9:
        best_fitness_ever = population[0]["fitness"]
        stagnant_generations = 0
        # Fresh improvement found — the streak that produced it is over,
        # so the anti-repetition counters reset: whatever family just
        # worked (or didn't) shouldn't be held against the NEXT streak.
        family_attempt_counts = {}
        recent_docstrings = []
    else:
        stagnant_generations += 1

    history.append({"generation": gen, "best_fitness": population[0]["fitness"],
                     "best_mean_margin": population[0]["mean_margin"],
                     "best_win_rate": population[0]["win_rate"],
                     "best_domain_scores": population[0]["domain_scores"],
                     "best_domain_zs": population[0]["domain_zs"],
                     "n_llm_failures_this_gen": sum(
                         (not c["used_llm"]) for c in children),
                     "n_rank_equivalent_duplicates": n_duplicate,
                     "stagnant_generations": stagnant_generations})
    if stagnant_generations >= 2 and family_attempt_counts:
        counts_str = ", ".join(f"{k}={v}" for k, v in sorted(family_attempt_counts.items()))
        anneal_note = f"  [stagnant={stagnant_generations}, annealing, tried:{{{counts_str}}}]"
    elif stagnant_generations >= 2:
        anneal_note = f"  [stagnant={stagnant_generations}, annealing]"
    else:
        anneal_note = ""
    jitter_note = f"  [jitter_streak={jitter_champion_streak}]" if jitter_champion_streak else ""
    print(f"gen {gen}: best fitness={population[0]['fitness']:.4f} "
          f"avg_z={population[0]['mean_margin']:+.4f} "
          f"domains_passed={population[0]['win_rate']:.2f} "
          f"domain_scores={population[0]['domain_scores']} "
          f"domain_zs={population[0]['domain_zs']} "
          f"({population[0]['id']})  duplicates={n_duplicate}/{len(children)}"
          f"{anneal_note}{jitter_note}")

    return (population, history, stagnant_generations, best_fitness_ever, n_llm_calls,
            n_llm_failures, family_attempt_counts, recent_docstrings, jitter_champion_streak)


def _print_llm_summary(n_llm_calls, n_llm_failures):
    print(f"\nReal-LLM calls: {n_llm_calls}, fell back to jittered-champion: "
          f"{n_llm_failures} ({100 * n_llm_failures / max(1, n_llm_calls):.0f}%)")
    if n_llm_failures == n_llm_calls:
        print("=> EVERY child fell back — this run used ZERO real LLM-authored "
              "code. Check the warnings above for the actual exception before "
              "trusting any result from this run.")
    elif n_llm_failures > 0:
        print("=> Some children fell back to a jittered champion copy — see "
              "warnings above for which calls failed and why.")


def main():
    ap = argparse.ArgumentParser()
    # Gate order IS this list's order — ZDT1 -> DTLZ2-3 -> ZDT3, ascending
    # noise (15%/30%/50% CV, see v6/README.md), cheapest/cleanest-read
    # domain first, same funnel-efficiency reasoning v5 used. Directory
    # defaults point at generate_noisy_synthetic_training_set.py's own
    # default output layout for each domain's train/ split.
    ap.add_argument("--zdt1_train_dir",
                     default=str(HERE.parent / "experiments" / "training_logs_zdt1" / "train"))
    ap.add_argument("--dtlz2_3obj_train_dir",
                     default=str(HERE.parent / "experiments" /
                                 "training_logs_dtlz2_3obj" / "train"))
    ap.add_argument("--zdt3_train_dir",
                     default=str(HERE.parent / "experiments" / "training_logs_zdt3" / "train"))
    ap.add_argument("--n_campaigns_zdt1", type=int, default=16,
                     help="Synthetic domains are cheap and reproducible (see "
                          "v6/README.md's substrate decision), so campaign counts "
                          "here are set higher than v5's real-domain defaults "
                          "(8/8/16) by default — directly targeting the underpowering "
                          "the diagnostic found (only 1 of 8 v5 'improvements' across "
                          "300 generations was distinguishable from noise).")
    ap.add_argument("--n_campaigns_dtlz2_3obj", type=int, default=16)
    ap.add_argument("--n_campaigns_zdt3", type=int, default=16)
    ap.add_argument("--pop_size", type=int, default=8)
    ap.add_argument("--n_generations", type=int, default=20)
    ap.add_argument("--n_offspring", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=GAMMA_V6_DEFAULT,
                     help="Complexity penalty, applied ONCE on the final combined "
                          "multi-domain fitness (not per-domain — see "
                          "evaluate_af_multi_domain's docstring). Defaults to 0.")
    ap.add_argument("--gate_z_threshold", type=float, default=None,
                     help="Per-domain z-score (ci_lower_16/se) threshold below which "
                          "a candidate is eliminated (fails that domain's gate) rather "
                          "than proceeding to the next domain. Defaults to "
                          "multi_domain_fitness_v6.GATE_Z_THRESHOLD (-1.0, i.e. 'at "
                          "least 1 SE below baseline') — see that module's docstring "
                          "for why v5's raw-margin epsilon was replaced with a "
                          "z-scored one.")
    ap.add_argument("--model", default="qwen3-coder:30b",
                     help="Ollama model for crossover/mutation and generation-0 hint "
                          "bootstrap — v6 is real-LLM-only, no --mock/--real_llm flag "
                          "(see af_interface_v6.py's module docstring for why).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective_names", default=None,
                     help="Comma-separated objective names shown in LLM prompts. "
                          "Defaults to a deliberately GENERIC placeholder (see "
                          "_MULTI_DOMAIN_DESCRIPTION) since real objective names "
                          "differ per domain and showing any ONE domain's names "
                          "would misleadingly anchor the model on it.")
    ap.add_argument("--domain_description", default=None,
                     help="Plain-English description of the domain(s) this run is "
                          "training against. Defaults to _MULTI_DOMAIN_DESCRIPTION "
                          "(generic, tells the model not to assume a fixed shape) "
                          "if not given.")
    ap.add_argument("--out_dir", default=str(HERE / "evolution_runs" / "run_v6"))
    ap.add_argument("--resume", action="store_true",
                     help="Continue a previous run from <out_dir>/checkpoint.json instead "
                          "of starting a fresh population. --n_generations is the TOTAL "
                          "target generation (not 'N more') — e.g. run once with "
                          "--n_generations 3 to see gen 0-3, inspect best_af.py, then rerun "
                          "the SAME command with --resume --n_generations 10 added to "
                          "continue on to generation 10. Requires --out_dir and every "
                          "--*_train_dir/--n_campaigns_*/--seed to match the original run "
                          "— training_logs/baseline_hvs are recomputed fresh each time "
                          "(cheap, not cached in the checkpoint) and must select the SAME "
                          "campaigns for fitness comparisons across generations to stay "
                          "meaningful; a checkpoint is written after EVERY generation "
                          "(including gen 0), always, on any run — not something you have "
                          "to opt into.")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_log_dir = out_dir / "af_code_logs"

    objective_names = ([n.strip() for n in args.objective_names.split(",")]
                        if args.objective_names else None)
    domain_description = args.domain_description  # None falls back to
                                                     # _MULTI_DOMAIN_DESCRIPTION
                                                     # inside run_evolution

    rng = np.random.default_rng(args.seed)

    # Domain order IS the gate sequence — ZDT1, DTLZ2-3, ZDT3 (ascending
    # noise: 15%/30%/50% CV, cheapest/cleanest-read first). See
    # evaluate_af_multi_domain's docstring / v6/README.md's "Domains"
    # section. n_fitness_seeds=1 for all three: unlike v5's real domains,
    # v6's synthetic substrate has no separate GP-fit-seed-noise diagnosis
    # calling for repeat-averaging (see mab_noise_diagnostic.py, which is
    # specific to excipient/mAb) — start simple, add it only if a future
    # diagnostic shows a need.
    domain_specs = [
        ("zdt1", args.zdt1_train_dir, args.n_campaigns_zdt1, "zdt1", 1),
        ("dtlz2_3obj", args.dtlz2_3obj_train_dir, args.n_campaigns_dtlz2_3obj, "dtlz2", 1),
        ("zdt3", args.zdt3_train_dir, args.n_campaigns_zdt3, "zdt3", 1),
    ]
    domain_configs = {}
    for domain_name, train_dir, n_campaigns, oracle_family, n_fitness_seeds in domain_specs:
        logs = load_training_campaigns(pathlib.Path(train_dir), n_campaigns, rng)
        print(f"Loaded {len(logs)} training campaigns for {domain_name} "
              f"(oracle_family={oracle_family}).")
        if not logs:
            print(f"No training campaigns found for {domain_name} at {train_dir} — "
                  f"generate them first with generate_noisy_synthetic_training_set.py "
                  f"(see v6/README.md for the exact per-domain noise_cv assignment).")
            return
        baseline_hvs = compute_baseline_hvs(logs, oracle_family=oracle_family,
                                             n_fitness_seeds=n_fitness_seeds)
        domain_configs[domain_name] = {
            "training_logs": logs, "baseline_hvs": baseline_hvs,
            "oracle_family": oracle_family, "n_fitness_seeds": n_fitness_seeds,
            # "directions" is per-domain (objective count/min-max convention
            # differs by domain) — see run_evolution's own comment on why
            # this must NOT be a single top-level default.
            "directions": ORACLE_DEFAULTS[oracle_family]["directions"],
        }

    checkpoint_path = out_dir / "checkpoint.json"
    resume_state = None
    if args.resume:
        if not checkpoint_path.exists():
            print(f"--resume given but no checkpoint found at {checkpoint_path} — "
                  f"nothing to resume from. Run without --resume first.")
            return
        resume_state = load_checkpoint(checkpoint_path)

    result = run_evolution(
        domain_configs, args.pop_size, args.n_generations, args.n_offspring,
        args.gamma, args.model, args.seed,
        objective_names=objective_names, domain_description=domain_description,
        log_dir=code_log_dir,
        gate_z_threshold=args.gate_z_threshold,
        checkpoint_path=checkpoint_path,
        resume_state=resume_state,
    )

    best = result["population"][0]
    print(f"\nBest AF: fitness={best['fitness']:.4f} avg_z={best['mean_margin']:+.4f} "
          f"domains_passed={best['win_rate']:.2f} "
          f"domain_scores={best['domain_scores']} domain_zs={best['domain_zs']} "
          f"LOC={best['loc']} ({best['id']})")
    print(f"\n{best['code']}")

    with open(out_dir / "best_af.py", "w") as f:
        f.write(best["code"])
    with open(out_dir / "history.json", "w") as f:
        json.dump({"history": result["history"],
                    "n_llm_calls": result["n_llm_calls"],
                    "n_llm_failures": result["n_llm_failures"],
                    "baseline_hvs": baseline_hvs}, f, indent=2)
    with open(out_dir / "final_population.json", "w") as f:
        json.dump(result["population"], f, indent=2)

    print(f"\nSaved best AF, history, and final population to {out_dir}")


if __name__ == "__main__":
    main()
