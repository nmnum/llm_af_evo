"""
evolve_af_v2.py — variant of evolve_af.py using af_interface_v2.py's
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
    python evolve_af_v2.py --train_dir training_logs/train --n_campaigns 8 \\
        --pop_size 8 --n_generations 20 --n_offspring 2 --mock \\
        --out_dir evolution_runs/run_v2
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
):
    sys.path.insert(0, str(_p))

from af_interface_v2 import (SEED_PROGRAMS, SEED_TERM_WEIGHTS, STRATEGY_HINTS,
                              count_loc, extract_af_docstring)
from fitness_common import to_allmax
from full_replay import run_2b_campaign, run_baseline_campaign
from mock_mutator import random_program, mutate, crossover, render_program

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
N_FITNESS_SEEDS_DEFAULTS = {"excipient": 3, "coatings": 1}

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
                                       oracle_family=oracle_family)["final_hv"]
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
    loc = count_loc(code)

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

    if gamma is None:
        se = float(np.std(rel_margins, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        gamma = se / MAX_LOC_SPREAD

    fitness = ci_lower_16 - gamma * loc

    return {"win_rate": win_rate, "mean_margin": mean_margin,
            "median_margin": median_margin, "ci_lower_16": ci_lower_16,
            "bootstrap_ci": bootstrap_ci, "gamma": gamma,
            "loc": loc, "fitness": fitness,
            "mean_hv": float(np.mean(hvs)) if hvs else float("nan"),
            "n_campaigns": n, "selection_signature": sig_hasher.hexdigest(),
            "docstring": extract_af_docstring(code)}


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
    names = list(objective_names)
    gp_lines = "\n".join(
        f'             "{n}": {{"mean": float, "std": float}},  # higher is better, already flipped if needed'
        for n in names)
    range_dict = ", ".join(f'"{n}": float' for n in names)

    return f'''You are an expert in {domain_description} and in Bayesian \
optimisation, evolving acquisition functions for a multi-objective \
Bayesian optimisation loop in this domain. You must write a Python function:

def score_pool(context) -> list:
    ...
    return scores  # one score per entry in context["pool"], higher = more preferred

REQUIRED — the first statement in your function body must be a one-line \
docstring in plain English describing what your AF does, e.g.:

def score_pool(context):
    """<one line: what this AF does, in plain English>"""
    ...

This is not optional decoration: score_pool programs that don't start with \
a non-empty one-line docstring are rejected outright before ever being run \
or scored. The sentence should describe the STRATEGY (e.g. "exploit near \
the front early, add an uncertainty penalty once stagnant"), not restate \
the code line-by-line.

context = {{
    "pool": [
        {{"x": np.ndarray({feature_dim},),               # normalised candidate features
         "gp_posterior": {{
{gp_lines}
         }}}},
        ...  # one entry per candidate
    ],
    "X_obs": np.ndarray(n_obs, {feature_dim}),          # every candidate observed so far
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
objective set, since the same score_pool contract is reused across different domains.

THE BASELINE YOU ARE TRYING TO BEAT: the current production method
("EGBO-novelty") does NOT use a linear combination of mean and std. Each
batch it: (1) fits a GP per objective, (2) proposes candidates by
optimising an acquisition score that estimates how much a candidate would
expand the dominated hypervolume, integrated over the GP's posterior
uncertainty (Monte Carlo sampled), (3) generates more candidates via an
evolutionary algorithm seeded from the current Pareto front, (4) scores
the combined pool with that same hypervolume-improvement estimate, then
(5) picks a batch using a rule that also rewards candidates far from
previously-observed points (novelty), not just high acquisition value.
Your score_pool must beat this — a plain mean+std combination is a much
simpler, weaker strategy than what you are actually competing against,
and simplicity is only rewarded here if it doesn't cost win rate.

WHAT pareto_front AND ref_point ARE FOR: they are provided so you can
reason about hypervolume improvement directly — how much a candidate would
expand the region of objective space that's better than every current
non-dominated point, if its predicted objectives are correct. A candidate
whose predicted objectives fall outside (beyond) the current front in some
direction, and far from where the front already reaches, expands the
dominated region more than one that's near the front or dominated by it
(inside the region the front already covers). front_range gives the
current front's per-objective spread if you want to normalise. You are not
required to use these fields — but the baseline's acquisition function is
built entirely around this idea, so an AF that ignores pareto_front/
ref_point is discarding the information the thing you're competing
against relies on most.

WHAT YOUR SCORE IS MEASURED BY: your fitness is the MEAN RELATIVE
IMPROVEMENT in final hypervolume across FULL TRAINING CAMPAIGNS (not
individual decision points, and not just a win/loss count) — using your
score_pool as the acquisition function for every batch of a real
sequential run, how much better (or worse) is your final hypervolume than
EGBO-novelty's own real final hypervolume on that same campaign, averaged
across campaigns. fitness = mean_relative_improvement - (a small penalty
per line of code). This means a choice that looks good for one batch but
leaves LATER batches worse off (e.g. by never exploring) is punished
directly by this fitness, not just caught by a separate later check —
consider how your choices at one decision compound into later ones, not
just whether this one decision looks good in isolation. It also means
margin matters: beating the baseline by a wide margin scores meaningfully
better than barely beating it, unlike a simple win/loss count.

HOW YOUR SCORES ARE USED: after score_pool returns, the top batch_size
candidates BY SCORE are taken directly as the batch — there is no
additional filtering, diversity, or hypervolume-maximisation step after
your scores are computed. Whatever you rank highest is what gets picked,
exactly as ranked. This means: if you want the batch to be diverse, your
scores themselves have to produce that (e.g. via a novelty term) — nothing
downstream will do it for you. It also means ranking accuracy matters most
near the top of the pool (the candidates that could plausibly make the
cut), not uniformly across every candidate.

Worked example — a simple exploitation-plus-uncertainty AF:

def score_pool(context):
    """Sum of predicted GP means plus normalised uncertainty (UCB-style)."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        scores.append(mu_sum + 2.0 * sigma_norm)
    return scores

This favours candidates with high predicted objectives (mu_sum) AND high
normalised uncertainty (sigma_norm) — a UCB-style tradeoff. You can do
better: e.g. use context["X_obs"] to add a novelty term (distance from
cand["x"] to the nearest observed point), or shift the RELATIVE WEIGHT
between mu_sum and sigma_norm as context["campaign"]["progress"] changes
(e.g. explore-weighted early, exploit-weighted late).

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
    score_pool implementing a one-line STRATEGY_HINT, rather than shipping
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
    from af_interface import has_return_statement

    system_prompt = _build_system_prompt(objective_names, domain_description, feature_dim)
    prompt = (
        f"Implement the following acquisition strategy as score_pool:\n\n"
        f"\"{hint}\"\n\n"
        f"Write a complete, correct score_pool(context) function for exactly "
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
                     "seed": int(rng.integers(1_000_000))},
        )
        code = resp["message"]["content"].strip()
        for fence in ["```python", "```"]:
            if code.startswith(fence):
                code = code[len(fence):]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    code = _generate(temperature=0.3, repeat_penalty=1.3)
    if not has_return_statement(code):
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


def llm_propose_child(parent_a: dict, parent_b: dict, best_so_far: dict, steps: list,
                       model: str, rng: np.random.Generator,
                       objective_names: list, domain_description: str,
                       feature_dim: int = 16, stagnant_generations: int = 0) -> str:
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
    """
    import ollama
    system_prompt = _build_system_prompt(objective_names, domain_description, feature_dim)
    diag = _campaign_diagnostic_text(steps)
    best_doc = best_so_far.get("docstring") or "(no docstring recorded)"
    prompt = (
        f"Best-so-far in the population (\"{best_so_far['id']}\", "
        f"mean_margin={best_so_far['mean_margin']:+.1%}): \"{best_doc}\"\n\n"
        f"Parent A (mean_margin={parent_a['mean_margin']:+.1%}, LOC={parent_a['loc']}):\n"
        f"```python\n{parent_a['code']}\n```\n\n"
        f"Parent B (mean_margin={parent_b['mean_margin']:+.1%}, LOC={parent_b['loc']}):\n"
        f"```python\n{parent_b['code']}\n```\n\n"
        f"Representative training campaign summary:\n{diag}\n\n"
        f"Write a new score_pool that combines or improves on these two parents. "
        f"Remember the required one-line docstring as the first statement."
    )

    # Annealing: only kicks in once stagnation is established (threshold 2
    # — gen 0-1 improving normally shouldn't trigger this), scales with
    # how long it's persisted, capped so it never becomes pure noise.
    STAGNATION_THRESHOLD = 2
    temperature = min(1.2, 0.7 + 0.06 * max(0, stagnant_generations - STAGNATION_THRESHOLD))
    if stagnant_generations >= STAGNATION_THRESHOLD:
        prompt += (
            f"\n\nNOTE: the population's best fitness has not improved for "
            f"{stagnant_generations} generations — small variations on the "
            f"current best (different fixed/decaying uncertainty weights, "
            f"minor novelty tweaks) have stopped working. Do NOT propose "
            f"another small weight variation this time. Instead, try a "
            f"STRUCTURALLY different mechanism from what's shown above — "
            f"e.g. diversity/repulsion between top-ranked candidates, "
            f"greedy proximity-based suppression of near-duplicates, "
            f"resampling the observed history to estimate improvement "
            f"under noise rather than trusting the current front as fixed, "
            f"or estimating each candidate's probability of being "
            f"Pareto-optimal via sampling from its own posterior — pick "
            f"ONE such mechanism and implement it as the dominant idea, "
            f"not as a minor addition to the mu_sum+uncertainty pattern "
            f"above."
        )

    resp = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_predict": 1024,
                 "repeat_penalty": 1.3,
                 "seed": int(rng.integers(1_000_000))},
    )
    code = resp["message"]["content"].strip()
    for fence in ["```python", "```"]:
        if code.startswith(fence):
            code = code[len(fence):]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


# ── Evolution loop ──────────────────────────────────────────────────────────

def tournament_select(population: list, rng: np.random.Generator, k: int = 3) -> dict:
    idx = rng.choice(len(population), size=min(k, len(population)), replace=False)
    contenders = [population[i] for i in idx]
    return max(contenders, key=lambda p: p["fitness"])


def make_child(parent_a: dict, parent_b: dict, best_so_far: dict, steps: list,
               mock: bool, model: str, rng: np.random.Generator,
               objective_names: list, domain_description: str, feature_dim: int = 16,
               stagnant_generations: int = 0):
    """Identical mechanism to evolve_af.py's make_child, plus annealing
    under stagnation — see llm_propose_child's docstring. mock mode has no
    LLM to anneal a prompt for, so it gets a light structural analog
    instead: force BOTH a crossover and a mutation (rather than a coin-
    flip mutation) once stagnation is established, for a bigger per-child
    perturbation than the default."""
    if mock:
        tw_a = parent_a.get("term_weights") or random_program(rng)
        tw_b = parent_b.get("term_weights") or random_program(rng)
        child_tw = crossover(tw_a, tw_b, rng)
        if stagnant_generations >= 2 or rng.random() < 0.5:
            child_tw = mutate(child_tw, rng)
        return render_program(child_tw), child_tw, True
    else:
        try:
            code = llm_propose_child(parent_a, parent_b, best_so_far, steps, model, rng,
                                      objective_names, domain_description, feature_dim,
                                      stagnant_generations=stagnant_generations)
            return code, None, True
        except Exception as e:
            warnings.warn(
                f"llm_propose_child failed ({type(e).__name__}: {e}) — "
                f"falling back to mock crossover for this child.\n"
                + traceback.format_exc())
            tw_a = parent_a.get("term_weights") or random_program(rng)
            tw_b = parent_b.get("term_weights") or random_program(rng)
            return render_program(crossover(tw_a, tw_b, rng)), None, False


def run_evolution(training_logs: list, baseline_hvs: list, pop_size: int,
                   n_generations: int, n_offspring: int, gamma: float,
                   mock: bool, model: str, seed: int,
                   objective_names: list = None, domain_description: str = None,
                   feature_dim: int = 16, directions: list = None,
                   oracle_family: str = "excipient",
                   log_dir: pathlib.Path = None,
                   n_fitness_seeds: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    objective_names = objective_names or ORACLE_DEFAULTS["excipient"]["objective_names"]
    domain_description = domain_description or ORACLE_DEFAULTS["excipient"]["domain_description"]
    pseudo_steps = build_pseudo_steps(training_logs, baseline_hvs, directions=directions)

    population = []
    for name, code in SEED_PROGRAMS.items():
        result = evaluate_af_2b(code, training_logs, baseline_hvs, gamma, log_dir=log_dir,
                                 oracle_family=oracle_family,
                                 n_fitness_seeds=n_fitness_seeds)
        population.append({"id": name, "code": code,
                            "term_weights": SEED_TERM_WEIGHTS.get(name), **result})

    if not mock:
        # Generation-0 bootstrap: ask the LLM to write score_pool for each
        # one-line STRATEGY_HINT, rather than shipping a hand-written
        # implementation for every strategy (see af_interface_v2.py's
        # module docstring). Mock mode has no LLM to turn a hint into
        # code, so it never touches STRATEGY_HINTS — it uses only
        # SEED_PROGRAMS + random-term-weight padding below.
        for hint_name, hint_text in STRATEGY_HINTS.items():
            try:
                code = llm_generate_seed_from_hint(hint_text, model, rng,
                                                    objective_names, domain_description,
                                                    feature_dim)
                term_weights = None
            except Exception as e:
                warnings.warn(
                    f"llm_generate_seed_from_hint failed for {hint_name!r} "
                    f"({type(e).__name__}: {e}) — falling back to a random "
                    f"mock program for this seed.\n" + traceback.format_exc())
                term_weights = random_program(rng)
                code = render_program(term_weights)
            result = evaluate_af_2b(code, training_logs, baseline_hvs, gamma, log_dir=log_dir,
                                     oracle_family=oracle_family,
                                     n_fitness_seeds=n_fitness_seeds)
            population.append({"id": f"hint_{hint_name}", "code": code,
                                "term_weights": term_weights, **result})

    # Pad with random mock programs for initial diversity beyond the
    # hand-written/hint-generated seeds, even in real-LLM mode.
    for i in range(max(0, pop_size - len(population))):
        tw = random_program(rng)
        code = render_program(tw)
        result = evaluate_af_2b(code, training_logs, baseline_hvs, gamma, log_dir=log_dir,
                                 oracle_family=oracle_family,
                                 n_fitness_seeds=n_fitness_seeds)
        population.append({"id": f"random_init_{i}", "code": code,
                            "term_weights": tw, **result})
    population = sorted(population, key=lambda p: -p["fitness"])[:max(pop_size, len(population))]

    history = [{"generation": 0, "best_fitness": population[0]["fitness"],
                "best_mean_margin": population[0]["mean_margin"],
                "best_win_rate": population[0]["win_rate"],
                "best_mean_hv": population[0]["mean_hv"]}]
    print(f"gen 0: best fitness={population[0]['fitness']:.4f} "
          f"mean_margin={population[0]['mean_margin']:+.3%} "
          f"win_rate={population[0]['win_rate']:.3f} mean_hv={population[0]['mean_hv']:.1f} "
          f"({population[0]['id']})")

    n_llm_calls, n_llm_failures = 0, 0
    best_fitness_ever = population[0]["fitness"]
    stagnant_generations = 0
    for gen in range(1, n_generations + 1):
        best_so_far = max(population, key=lambda p: p["fitness"])
        children = []
        for i in range(n_offspring):
            parent_a = tournament_select(population, rng)
            parent_b = tournament_select(population, rng)
            code, term_weights, used_llm = make_child(parent_a, parent_b, best_so_far,
                                                        pseudo_steps, mock, model, rng,
                                                        objective_names, domain_description,
                                                        feature_dim,
                                                        stagnant_generations=stagnant_generations)
            if not mock:
                n_llm_calls += 1
                n_llm_failures += (not used_llm)
            result = evaluate_af_2b(code, training_logs, baseline_hvs, gamma, log_dir=log_dir,
                                     oracle_family=oracle_family,
                                     n_fitness_seeds=n_fitness_seeds)
            children.append({"id": f"gen{gen}_child{i}", "code": code,
                              "term_weights": term_weights, "used_llm": used_llm, **result})

        existing_sigs = {p["selection_signature"] for p in population}
        novel_children, n_duplicate = [], 0
        for c in children:
            if c["selection_signature"] in existing_sigs:
                n_duplicate += 1
                continue
            existing_sigs.add(c["selection_signature"])
            novel_children.append(c)

        population = sorted(population + novel_children, key=lambda p: -p["fitness"])[:pop_size]

        # Stagnation tracking for next generation's annealing (see
        # llm_propose_child's docstring) — small epsilon tolerance so
        # floating-point-noise-level "improvement" doesn't reset the
        # counter and mask genuine stagnation.
        if population[0]["fitness"] > best_fitness_ever + 1e-9:
            best_fitness_ever = population[0]["fitness"]
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        history.append({"generation": gen, "best_fitness": population[0]["fitness"],
                         "best_mean_margin": population[0]["mean_margin"],
                         "best_win_rate": population[0]["win_rate"],
                         "best_mean_hv": population[0]["mean_hv"],
                         "n_llm_failures_this_gen": sum(
                             (not c["used_llm"]) for c in children),
                         "n_rank_equivalent_duplicates": n_duplicate,
                         "stagnant_generations": stagnant_generations})
        anneal_note = f"  [stagnant={stagnant_generations}, annealing]" if stagnant_generations >= 2 else ""
        print(f"gen {gen}: best fitness={population[0]['fitness']:.4f} "
              f"mean_margin={population[0]['mean_margin']:+.3%} "
              f"win_rate={population[0]['win_rate']:.3f} mean_hv={population[0]['mean_hv']:.1f} "
              f"({population[0]['id']})  duplicates={n_duplicate}/{len(children)}{anneal_note}")

    if not mock:
        print(f"\nReal-LLM calls: {n_llm_calls}, fell back to mock crossover: "
              f"{n_llm_failures} ({100 * n_llm_failures / max(1, n_llm_calls):.0f}%)")
        if n_llm_failures == n_llm_calls:
            print("=> EVERY child fell back — this run used ZERO real LLM-authored "
                  "code, despite --real_llm. Check the warnings above for the "
                  "actual exception before trusting any result from this run.")
        elif n_llm_failures > 0:
            print("=> Some children fell back to mock crossover — see warnings "
                  "above for which calls failed and why.")

    return {"population": population, "history": history,
            "n_llm_calls": n_llm_calls, "n_llm_failures": n_llm_failures}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default=str(HERE / "training_logs" / "train"))
    ap.add_argument("--n_campaigns", type=int, default=8,
                     help="number of training campaigns to replay per candidate AF "
                          "evaluation — full-campaign replay is far more expensive "
                          "per candidate than the old per-step proxy, so this and "
                          "pop_size/n_offspring below default to evolve_af_2b.py's "
                          "already cost-validated smaller values, not evolve_af_v2's "
                          "original (2a-proxy-calibrated) 16/8 defaults.")
    ap.add_argument("--pop_size", type=int, default=8)
    ap.add_argument("--n_generations", type=int, default=20)
    ap.add_argument("--n_offspring", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=None,
                     help="Complexity penalty. Leave unset (default) for the "
                          "ADAPTIVE gamma evaluate_af_2b computes per call: "
                          "gamma = SE(mean_margin) / MAX_LOC_SPREAD, measured "
                          "fresh from each candidate's own rel_margins — this "
                          "caps the total possible LOC penalty at one standard "
                          "error, so LOC can only break ties between AFs whose "
                          "fitness is within noise distance of each other, never "
                          "override a genuine performance gap. See "
                          "evaluate_af_2b's GAMMA CALIBRATION docstring. Pass an "
                          "explicit value here only to reproduce an old run's "
                          "fixed-gamma behavior.")
    ap.add_argument("--mock", action="store_true", default=True,
                     help="Use mock (term-weight) crossover instead of an LLM. NOTE: "
                          "this does NOT make evaluation cheap — every candidate, mock "
                          "or real-LLM, still runs full real-campaign replay (GP fits, "
                          "optimize_acqf, evolutionary candidate generation) via "
                          "full_replay.run_2b_campaign. --mock only skips the LLM call "
                          "used for crossover/mutation itself; it is a correctness/"
                          "pipeline sanity check, not a cheap dry run.")
    ap.add_argument("--real_llm", dest="mock", action="store_false")
    ap.add_argument("--model", default="qwen3-coder:30b",
                     help="Ollama model for --real_llm crossover/mutation and "
                          "generation-0 hint bootstrap.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle", choices=list(ORACLE_DEFAULTS.keys()), default="excipient",
                     help="Which oracle family this run trains against. Selects the "
                          "oracle full_replay.run_2b_campaign/run_baseline_campaign "
                          "reconstruct (see full_replay.reconstruct_oracle's "
                          "oracle_family parameter) AND derives default "
                          "objective_names/domain_description/feature_dim for that "
                          "family (ORACLE_DEFAULTS) — each individually overridable "
                          "below. 'coatings' requires training-log JSON files "
                          "generated by generate_coatings_training_set.py, not the "
                          "excipient logs generate_training_set.py produces.")
    ap.add_argument("--objective_names", default=None,
                     help="Comma-separated objective names. Defaults to "
                          "ORACLE_DEFAULTS[--oracle]['objective_names'] if not given.")
    ap.add_argument("--domain_description", default=None,
                     help="Plain-English description of the domain this run is "
                          "training against, used to frame the LLM as a domain "
                          "expert. Defaults to ORACLE_DEFAULTS[--oracle]"
                          "['domain_description'] if not given.")
    ap.add_argument("--n_fitness_seeds", type=int, default=None,
                     help="Number of (GP-fit, UNSGA3) seed repeats to average per "
                          "campaign when computing fitness. Defaults to "
                          "N_FITNESS_SEEDS_DEFAULTS[--oracle] if not given (3 for "
                          "excipient/mAb, 1 for coatings — see that dict's "
                          "docstring). This addresses SEED noise specifically "
                          "(mab_noise_diagnostic.py measured a paired-margin "
                          "seed-to-seed std of 0.2557 on one fixed mAb campaign, "
                          "larger than either raw HV series's own seed variance) "
                          "but is NOT sufficient alone — domain/campaign-to-"
                          "campaign noise is still the majority of mAb's total "
                          "variance and still needs the bootstrap-CI-lower-bound "
                          "fitness and/or more --n_campaigns regardless of this "
                          "setting. Directly "
                          "multiplies the cost of every fitness evaluation by this "
                          "factor (baseline compute too).")
    ap.add_argument("--out_dir", default=str(HERE / "evolution_runs" / "run_v2"))
    args = ap.parse_args()

    if args.gamma is None:
        print("--gamma not given, using the ADAPTIVE gamma computed fresh per "
              "candidate inside evaluate_af_2b (SE(mean_margin)/MAX_LOC_SPREAD).")

    if args.n_fitness_seeds is None:
        args.n_fitness_seeds = N_FITNESS_SEEDS_DEFAULTS[args.oracle]
        print(f"--n_fitness_seeds not given, using "
              f"N_FITNESS_SEEDS_DEFAULTS[{args.oracle!r}]={args.n_fitness_seeds}")

    if args.oracle == "excipient":
        print(
            "\n" + "=" * 70 +
            "\nNOTE: --oracle excipient (mAb). The training fitness signal is\n"
            "noise-dominated at typical training-set sizes: baseline_hv CV~50%\n"
            "at n=75 training campaigns gives SE(mean_margin)~0.057, while\n"
            "observed top-AF margin gaps are ~0.011 (~0.2 SE) — well within\n"
            "noise on a raw point estimate. Fitness now uses a bootstrap\n"
            "16th-percentile CI-lower-bound on the mean instead of the mean\n"
            "itself (see evaluate_af_2b's docstring) specifically to make this\n"
            "noise-vs-signal problem visible in the ranking rather than hidden\n"
            "behind a false-precision point estimate. --n_fitness_seeds="
            f"{args.n_fitness_seeds} averages out SEED-driven noise on top of "
            "that (mab_noise_diagnostic.py found seed noise accounts for ~27% "
            "of mAb's per-campaign margin variance, domain noise the rest) — "
            "still worth increasing --n_campaigns substantially before "
            "treating this run's winner as a real finding.\n" + "=" * 70 + "\n")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_log_dir = out_dir / "af_code_logs"

    defaults = ORACLE_DEFAULTS[args.oracle]
    objective_names = ([n.strip() for n in args.objective_names.split(",")]
                        if args.objective_names else defaults["objective_names"])
    domain_description = args.domain_description or defaults["domain_description"]
    feature_dim = defaults["feature_dim"]
    directions = defaults["directions"]

    rng = np.random.default_rng(args.seed)
    training_logs = load_training_campaigns(pathlib.Path(args.train_dir), args.n_campaigns, rng)
    print(f"Loaded {len(training_logs)} training campaigns (oracle={args.oracle}).")
    if not training_logs:
        print("No training campaigns found — run generate_training_set.py "
              "(excipient) or generate_coatings_training_set.py (coatings) first.")
        return

    baseline_hvs = compute_baseline_hvs(training_logs, oracle_family=args.oracle,
                                         n_fitness_seeds=args.n_fitness_seeds)

    result = run_evolution(
        training_logs, baseline_hvs, args.pop_size, args.n_generations, args.n_offspring,
        args.gamma, args.mock, args.model, args.seed,
        objective_names=objective_names, domain_description=domain_description,
        feature_dim=feature_dim, directions=directions, oracle_family=args.oracle,
        log_dir=code_log_dir,
        n_fitness_seeds=args.n_fitness_seeds,
    )

    best = result["population"][0]
    print(f"\nBest AF: fitness={best['fitness']:.4f} mean_margin={best['mean_margin']:+.3%} "
          f"win_rate={best['win_rate']:.3f} "
          f"mean_hv={best['mean_hv']:.1f} LOC={best['loc']} ({best['id']})")
    print(f"\n{best['code']}")

    with open(out_dir / "best_af.py", "w") as f:
        f.write(best["code"])
    with open(out_dir / "history.json", "w") as f:
        json.dump({"history": result["history"], "mock": args.mock,
                    "n_llm_calls": result["n_llm_calls"],
                    "n_llm_failures": result["n_llm_failures"],
                    "baseline_hvs": baseline_hvs}, f, indent=2)
    with open(out_dir / "final_population.json", "w") as f:
        json.dump([{k: v for k, v in p.items() if k != "term_weights"}
                    for p in result["population"]], f, indent=2)

    print(f"\nSaved best AF, history, and final population to {out_dir}")


if __name__ == "__main__":
    main()
