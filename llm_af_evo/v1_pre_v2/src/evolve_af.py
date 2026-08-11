"""
evolve_af.py — L1 build, step 4: FunBO-style programs-database evolution of
score_pool AFs (see af_interface.py's contract), fit against the fixed
training set (generate_training_set.py's output).

Fitness (per af_interface.py / fitness_common.py): for each training step,
run the candidate AF in the sandbox, select its batch (fixed top-k logic,
af_interface.select_batch), score that batch with the validated GP-only
exploration-credit metric (HV_pred_improvement + lambda*=10000 * sigma_norm,
no oracle), and compare against EGBO-novelty's own actual pick scored the
same way. fitness = mean_win_rate - gamma * LOC(code) — the complexity
penalty, doing double duty as (a) noise-robustness against the per-campaign
variance the sweep_lambda.py run exposed, and (b) an anti-rediscovery
pressure toward AFs that are structurally simpler than EGBO-novelty, not
just numerically equivalent to it under a different parameterisation.

Evolution loop: mu+lambda (elitist) — each generation, sample parents from
the current population via tournament selection, produce n_offspring
children (mock structural mutator or real LLM rewrite), evaluate them,
then keep the top pop_size programs from population UNION children.

Mock mode (default, and the only mode runnable in this sandbox — no
Ollama/network here) uses mock_mutator.py's deliberately weak,
uninformed term-weight crossover/mutation. Real-LLM mode is wired
(--mock=False --model=...) for you to run on your machine, following the
same ollama.chat pattern as strategy_mo_llm/llm_warmstart.py elsewhere in
this repo.

KNOWN GAP, flagged rather than silently skipped: the Harris group's
graphical diagnostics (a plot comparing an AF's picks to the true Pareto
front, shown to the LLM) are NOT implemented here — only a text/numeric
summary of a few representative training campaigns is included in the
real-LLM prompt. Building and validating an actual image-generation +
multimodal-chat path isn't possible to verify in this sandbox (no
matplotlib, no Ollama, no vision-capable model access here), so it's left
as an explicit follow-up rather than shipped unverified.

Usage:
    python evolve_af.py --train_dir training_logs/train --pop_size 16 \\
        --n_generations 20 --n_offspring 8 --mock --out_dir evolution_runs/run1
"""

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

from af_interface import (SEED_PROGRAMS, SEED_TERM_WEIGHTS, select_batch,
                           count_loc, extract_af_docstring)
from fitness_common import to_allmax, hv_of, true_hv_gain_of_pick
from sandbox import run_af_in_sandbox, SandboxError
from mock_mutator import random_program, mutate, crossover, render_program


# ── Training data loading ──────────────────────────────────────────────────

def _stagnant_prefix(hv_trajectory: list, upto_idx: int) -> int:
    """
    Count of non-improving consecutive batches within hv_trajectory[:upto_idx]
    — i.e. stagnation as it would have been VISIBLE to the AF at decision
    time for the batch at index upto_idx, not counting that batch's own
    (not-yet-known) outcome. Same non-improvement rule as
    run_benchmark_resumable.py's count_stagnant_batches, just restricted to
    the prefix — this is free, derived entirely from data every training
    log already saves (hv_trajectory), no new campaign instrumentation.
    """
    sub = hv_trajectory[:upto_idx]
    if len(sub) < 2:
        return 0
    return sum(1 for i in range(1, len(sub)) if sub[i] <= sub[i - 1] + 1e-6)


def load_training_steps(train_dir: pathlib.Path) -> list:
    """
    Flatten every campaign log's per-batch decisions into a list of
    self-contained step records, each with everything evaluate_af needs:
    the pool, the oracle pool (for scoring a CANDIDATE AF's hypothetical
    picks against ground truth), EGBO-novelty's own true HV gain at that
    step (computed directly from its logged picked_y — already real
    observations, no lookup needed), batch_size, and the campaign-state
    scalars (step, budget, n_obs, stagnant_batches) the AF interface now
    takes so evolved programs can express phase-dependent strategies.
    """
    steps = []
    for f in sorted(pathlib.Path(train_dir).glob("*.json")):
        with open(f) as fh:
            log = json.load(fh)

        Y_running = np.array(log["Y_init"])
        X_running = np.array(log["X_init"])
        oracle_X = np.array(log["oracle_X_raw"])
        oracle_Y = np.array(log["oracle_Y_raw"])
        Y_all_allmax = to_allmax(oracle_Y)
        ref_point_allmax = (Y_all_allmax.min(axis=0) -
                             0.1 * (Y_all_allmax.max(axis=0) - Y_all_allmax.min(axis=0) + 1e-9))
        batch_size = log["batch_size"]
        budget = log["budget"]
        hv_trajectory = log["hv_trajectory"]

        # objective_names: threaded through to run_af_in_sandbox so evolved
        # code sees the SAME keys a real deployment context would use for
        # this domain (see sandbox.py's run_af_in_sandbox docstring). Bug
        # found 2026-08-11: this used to be omitted entirely, so every
        # training step (mab AND dtlz2/zdt1) silently used
        # run_af_in_sandbox's ["Tm","kD","viscosity"] default regardless of
        # domain — evolution against dtlz2 data still produced correct
        # win_rate numbers (positionally consistent labels throughout
        # training), but the resulting best_af.py hardcoded mAb's names and
        # would KeyError immediately in a real dtlz2 campaign, where
        # context["gp_posterior"] is actually keyed by f1/f2/f3. mab logs
        # don't carry a "domain" key (pre-dates --domain), so absence of
        # that key still means mAb's Tm/kD/viscosity, matching
        # generate_coreg_steps.py's mab payload shape.
        domain = log.get("domain")
        if domain in ("dtlz2", "zdt1"):
            objective_names = [f"f{i + 1}" for i in range(Y_running.shape[1])]
        else:
            objective_names = ["Tm", "kD", "viscosity"]

        for batch_idx, step in enumerate(log["decisions"]):
            if "pool_x_norm" not in step or "pool_pred_sigma" not in step:
                Y_running = np.vstack([Y_running, np.array(step["picked_y"])]) \
                    if "picked_y" in step else Y_running
                continue

            front_allmax = to_allmax(Y_running.copy())
            range_j = np.maximum(np.ptp(Y_running, axis=0), 1e-6)
            pool_mu = np.array(step["pool_pred_mu"])
            pool_sigma = np.array(step["pool_pred_sigma"])

            # EGBO-novelty's true HV gain — picked_y is already a real
            # observation (logged by run_mo_campaign), so this needs no
            # oracle lookup, unlike a candidate AF's hypothetical picks.
            front_hv = hv_of(front_allmax, ref_point_allmax)
            actual_true_y = np.array(step["picked_y"])
            egbo_true_gain = hv_of(np.vstack([front_allmax, to_allmax(actual_true_y)]),
                                    ref_point_allmax) - front_hv

            steps.append({
                "campaign": f.name, "step": step["step"], "batch_size": batch_size,
                "budget": budget,
                "n_obs": step.get("n_obs_before_pick", step["step"]),
                "stagnant_batches": _stagnant_prefix(hv_trajectory, batch_idx),
                "pool_x": np.array(step["pool_x_norm"]),
                "pool_mu": pool_mu, "pool_sigma": pool_sigma,
                # {} for every pre-existing independent-GP training log
                # (generate_training_set.py never wrote this key) — only
                # generate_coreg_steps.py's DA-COREG campaigns populate it.
                "obj_correlation": step.get("pool_obj_correlation", {}),
                "X_obs": X_running.copy(), "Y_obs": front_allmax.copy(),
                "front_allmax": front_allmax, "ref_point_allmax": ref_point_allmax,
                "range_j": range_j,
                "oracle_X": oracle_X, "oracle_Y": oracle_Y,
                "egbo_true_gain": egbo_true_gain,
                "objective_names": objective_names,
            })

            Y_running = np.vstack([Y_running, np.array(step["picked_y"])])
            X_running = np.vstack([X_running, np.array(step["picked_x"])])

    return steps


# ── Fitness evaluation ──────────────────────────────────────────────────────

def evaluate_af(code: str, steps: list, gamma: float, log_dir=None) -> dict:
    """
    Run `code` against every training step in the sandbox. A sandbox
    failure (timeout, bad shape, non-finite, import violation) counts as a
    LOSS for that step, not an exclusion — otherwise a program that always
    crashes would trivially avoid ever losing, which is the opposite of
    what we want the fitness to reward.

    Scoring is against TRUE-ORACLE HV gain, not the GP-only exploration-
    credit proxy (see fitness_common.af_fitness_of_set's docstring for why
    that proxy is unsound as a training signal: it's dominated by the
    sigma term at lambda*=10000, so an AF that maximises uncertainty alone
    — ignoring predicted quality (mu) entirely — scores near-optimally on
    it regardless of whether its picks are any good. run1's evolved
    population converged to exactly that degenerate form. score_pool
    itself still only ever sees mu/sigma/x (never the oracle) — only the
    training-time SCORING of its output uses ground truth, which is
    legitimate for offline synthetic training (see true_hv_gain_of_pick's
    docstring).
    """
    if log_dir is not None:
        # Log the candidate ONCE per evaluate_af call, not once per training
        # step — passing log_dir into every per-step run_af_in_sandbox call
        # below wrote the same code to a new numbered file per step (360x
        # duplication per candidate), which is what produced 86,400 files
        # from a ~17,000-call run and made every sandbox call pay for an
        # O(n) glob("call_*.py") just to pick the next filename.
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(log_dir.glob("call_*.py")))
        (log_dir / f"call_{existing:05d}.py").write_text(code)

    # selection_signature: a hash of EVERY step's selected-batch indices,
    # not just win/loss. Two AFs that pick the identical batch on every
    # step are rank-equivalent — select_batch only depends on argsort
    # order, so any AF that's a positive-scalar-per-step reparametrisation
    # of another produces the same picks and therefore the same win_rate,
    # even if the source code looks structurally different. This is what
    # run1 turned out to be: 16 "different" evolved programs, all of the
    # form k(step) * sigma_term(cand) where k(step) is constant across the
    # pool at a given step (a function only of progress/stagnant_batches,
    # not of cand) — multiplying every candidate's score by the same
    # positive constant never changes which ones rank highest. Comparing
    # this signature lets run_evolution detect and discard children that
    # can't possibly add anything the population doesn't already have,
    # rather than burning evolutionary budget re-discovering them.
    wins, n_failed = 0, 0
    sig_hasher = hashlib.md5()
    for s in steps:
        try:
            scores = run_af_in_sandbox(
                code, s["pool_x"], s["pool_mu"], s["pool_sigma"], s["X_obs"],
                s["front_allmax"], s["ref_point_allmax"],
                s["step"], s["budget"], s["n_obs"], s["stagnant_batches"],
                s["Y_obs"], obj_correlation=s["obj_correlation"],
                objective_names=s["objective_names"],
            )
            af_idx = select_batch(scores, s["batch_size"])
            sig_hasher.update(str(sorted(af_idx)).encode())
            af_true_gain = true_hv_gain_of_pick(
                af_idx, s["pool_x"], s["oracle_X"], s["oracle_Y"],
                s["front_allmax"], s["ref_point_allmax"])
            if af_true_gain > s["egbo_true_gain"]:
                wins += 1
        except SandboxError:
            n_failed += 1
            sig_hasher.update(b"SANDBOX_FAILURE")

    n = len(steps)
    win_rate = wins / n if n else 0.0
    loc = count_loc(code)
    fitness = win_rate - gamma * loc
    return {"win_rate": win_rate, "loc": loc, "fitness": fitness,
            "n_steps": n, "n_sandbox_failures": n_failed,
            "selection_signature": sig_hasher.hexdigest(),
            "docstring": extract_af_docstring(code)}


# ── Real-LLM child generation (wired, untestable in this sandbox) ─────────

_LLM_SYSTEM_PROMPT = '''You are evolving acquisition functions for multi-objective \
Bayesian optimisation. The objectives being optimised vary by run (e.g. protein \
formulation properties, or synthetic benchmark objectives) — NEVER hardcode \
objective names; always read them from context["objective_names"] (see below), \
so the same score_pool works regardless of which domain it's being evolved for. \
You must write a Python function:

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

context = {
    "objective_names": [str, ...],           # e.g. ["Tm", "kD", "viscosity"] or ["f1", "f2", "f3"]
                                              # — ALWAYS iterate this, never hardcode names;
                                              # it tells you both the names AND how many
                                              # objectives this run has (not always 3).
    "pool": [
        {"x": np.ndarray(d,),                # normalised candidate features
         "gp_posterior": {
             # one entry per name in context["objective_names"], e.g.:
             # "Tm": {"mean": float, "std": float},  # ALREADY flipped: higher is always better
         }},
        ...  # one entry per candidate
    ],
    "X_obs": np.ndarray(n_obs, d),           # every point observed so far
    "Y_obs": np.ndarray(n_obs, len(objective_names)),  # every OUTCOME observed so far,
                                              # all-maximised, rows match X_obs, columns
                                              # match context["objective_names"] order
                                              # (full observation history, not filtered to
                                              # the front — use for novelty distance or
                                              # resampling a noisy Pareto front)
    "pareto_front": np.ndarray(n_pf, len(objective_names)),  # non-dominated points,
                                              # columns match context["objective_names"]
    "pareto_front_range": {name: float, ...},  # observed range per objective, keyed by name
    "ref_point": np.ndarray(len(objective_names),),  # HV reference point, all-maximised,
                                              # same column order as objective_names
    "ref_point_by_name": {name: float, ...},
    "campaign": {"step": int, "budget": int, "progress": float,  # step/budget, in [0,1]
                 "n_obs": int, "stagnant_batches": int},  # consecutive non-improving batches
}

Every objective in gp_posterior/pareto_front_range/ref_point_by_name is ALREADY flipped so
higher is always better, whatever the true direction (min/max) — do not flip it again.
Access objectives BY NAME via context["objective_names"] (e.g.
`for name in context["objective_names"]: gp[name]["std"]`), never by a hardcoded name or a
remembered array position — the exact same score_pool source must work whether this run has
3 objectives called Tm/kD/viscosity or 3 called f1/f2/f3.

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

WHAT YOUR SCORE IS MEASURED BY: during training, an AF's win_rate is the
fraction of individually-logged historical decision points where your
score_pool's top-batch_size picks would have increased the true Pareto
front's hypervolume MORE than what the baseline actually picked at that
same point (both measured against the real, not GP-predicted, outcome).
fitness = win_rate - (a small penalty per line of code). The FINAL
evaluation that decides whether an AF is actually good, however, runs your
score_pool as the acquisition function for complete, multi-batch campaigns
on new points and compares final hypervolume — so an AF that wins
many individual decision points but makes choices that leave later
decisions worse off (e.g. by never exploring) can still under-perform
end-to-end. Consider how your choices at one decision compound into later
ones, not just whether this one decision looks good in isolation.

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
    """
    Text/numeric stand-in for the Harris group's graphical diagnostics
    (see module docstring's KNOWN GAP note) — a few representative steps'
    numbers, not a plot.
    """
    sample = steps[:: max(1, len(steps) // n_campaigns)][:n_campaigns]
    lines = []
    for s in sample:
        lines.append(f"  {s['campaign']} step {s['step']}/{s['budget']} "
                      f"(stagnant_batches={s['stagnant_batches']}): front_size="
                      f"{len(s['front_allmax'])}, EGBO-novelty true HV gain="
                      f"{s['egbo_true_gain']:.2f}")
    return "\n".join(lines)


def llm_propose_child(parent_a: dict, parent_b: dict, best_so_far: dict, steps: list,
                       model: str, rng: np.random.Generator) -> str:
    """Real-LLM crossover/mutation — untestable here (no Ollama access)."""
    import ollama
    diag = _campaign_diagnostic_text(steps)
    # best_so_far (population[0], not necessarily either parent) is called
    # out on its own line, distinct from the two tournament-selected
    # parents' full code blocks below — this lets the model reason about
    # what the population as a whole has converged toward ("previous best
    # was 'pure exploitation near the front' — try adding an exploration
    # term for early batches") even when tournament selection didn't happen
    # to draw the elite as one of this child's own parents.
    best_doc = best_so_far.get("docstring") or "(no docstring recorded)"
    prompt = (
        f"Best-so-far in the population (\"{best_so_far['id']}\", "
        f"win_rate={best_so_far['win_rate']:.3f}): \"{best_doc}\"\n\n"
        f"Parent A (win_rate={parent_a['win_rate']:.3f}, LOC={parent_a['loc']}):\n"
        f"```python\n{parent_a['code']}\n```\n\n"
        f"Parent B (win_rate={parent_b['win_rate']:.3f}, LOC={parent_b['loc']}):\n"
        f"```python\n{parent_b['code']}\n```\n\n"
        f"Representative training campaign summary:\n{diag}\n\n"
        f"Write a new score_pool that combines or improves on these two parents. "
        f"Remember the required one-line docstring as the first statement."
    )
    resp = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": _LLM_SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        options={"temperature": 0.7, "num_predict": 512,
                 "seed": int(rng.integers(1_000_000))},
    )
    code = resp["message"]["content"].strip()
    for fence in ["```python", "```"]:
        if code.startswith(fence):
            code = code[len(fence):]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


# ── Evolution loop ───────────────────────────────────────────────────────────

def tournament_select(population: list, rng: np.random.Generator, k: int = 3) -> dict:
    idx = rng.choice(len(population), size=min(k, len(population)), replace=False)
    contenders = [population[i] for i in idx]
    return max(contenders, key=lambda p: p["fitness"])


def make_child(parent_a: dict, parent_b: dict, best_so_far: dict, steps: list,
               mock: bool, model: str, rng: np.random.Generator):
    """
    Returns (code, term_weights, used_llm). used_llm is True in mock mode
    (the mutator IS the intended mechanism there) and True in real-LLM mode
    only if llm_propose_child actually succeeded — False means it fell back
    to mock crossover after an exception, which is now logged loudly rather
    than silently, after a real run's output turned out to be
    indistinguishable from a pure mock run because every single
    llm_propose_child call was hitting the same bug (KeyError from a stale
    dict key after the true-oracle-fitness fix) and falling back unnoticed.
    """
    if mock:
        # Only well-defined if both parents carry term_weights (i.e. came
        # from the mock path themselves) — every mock-generated program
        # does, including the rendered seeds (seeded with term_weights
        # below in run_evolution), so this holds throughout a mock run.
        tw_a = parent_a.get("term_weights") or random_program(rng)
        tw_b = parent_b.get("term_weights") or random_program(rng)
        child_tw = crossover(tw_a, tw_b, rng)
        if rng.random() < 0.5:
            child_tw = mutate(child_tw, rng)
        return render_program(child_tw), child_tw, True
    else:
        try:
            code = llm_propose_child(parent_a, parent_b, best_so_far, steps, model, rng)
            return code, None, True
        except Exception as e:
            warnings.warn(
                f"llm_propose_child failed ({type(e).__name__}: {e}) — "
                f"falling back to mock crossover for this child.\n"
                + traceback.format_exc())
            tw_a = parent_a.get("term_weights") or random_program(rng)
            tw_b = parent_b.get("term_weights") or random_program(rng)
            return render_program(crossover(tw_a, tw_b, rng)), None, False


def run_evolution(steps: list, pop_size: int, n_generations: int, n_offspring: int,
                   gamma: float, mock: bool, model: str, seed: int,
                   log_dir: pathlib.Path = None, patience: int = 6) -> dict:
    rng = np.random.default_rng(seed)

    population = []
    for name, code in SEED_PROGRAMS.items():
        result = evaluate_af(code, steps, gamma, log_dir=log_dir)
        population.append({"id": name, "code": code,
                            "term_weights": SEED_TERM_WEIGHTS.get(name), **result})
    # Pad with a few random mock programs for initial diversity beyond the
    # 4 hand-written seeds, even in real-LLM mode (cheap, keeps the initial
    # population from being only 4 individuals).
    for i in range(max(0, pop_size - len(population))):
        tw = random_program(rng)
        code = render_program(tw)
        result = evaluate_af(code, steps, gamma, log_dir=log_dir)
        population.append({"id": f"random_init_{i}", "code": code,
                            "term_weights": tw, **result})

    history = [{"generation": 0,
                "best_fitness": max(p["fitness"] for p in population),
                "best_win_rate": max(p["win_rate"] for p in population)}]

    n_llm_calls, n_llm_failures = 0, 0
    no_improve_count = 0
    best_fitness_so_far = history[0]["best_fitness"]
    for gen in range(1, n_generations + 1):
        # Recomputed each generation rather than relying on population
        # being pre-sorted — it IS sorted after gen 1 (bottom of this loop
        # sorts it), but the initial population (seeds + random inits) is
        # only ever fitness-sorted starting here, so don't assume order.
        best_so_far = max(population, key=lambda p: p["fitness"])
        children = []
        for i in range(n_offspring):
            parent_a = tournament_select(population, rng)
            parent_b = tournament_select(population, rng)
            code, term_weights, used_llm = make_child(parent_a, parent_b, best_so_far,
                                                        steps, mock, model, rng)
            if not mock:
                n_llm_calls += 1
                n_llm_failures += (not used_llm)
            result = evaluate_af(code, steps, gamma, log_dir=log_dir)
            children.append({"id": f"gen{gen}_child{i}", "code": code,
                              "term_weights": term_weights, "used_llm": used_llm,
                              **result})

        # Rank-equivalence dedup: a child whose selection_signature matches
        # something already in the population picks the IDENTICAL batch on
        # every training step as that existing individual — it cannot add
        # any new information no matter how different its source code
        # looks, and keeping it just occupies a population slot and a
        # future tournament-selection draw with a redundant parent. This is
        # what let run1 spend all 160 real-LLM calls re-discovering
        # variants of the same k(step)*sigma_term(cand) ranking function
        # (see evaluate_af's selection_signature docstring) — 16 distinct
        # source strings, but only ever 1 distinct BEHAVIOUR in the
        # population, so nothing new could ever win a tournament.
        existing_sigs = {p["selection_signature"] for p in population}
        novel_children, n_duplicate = [], 0
        for c in children:
            if c["selection_signature"] in existing_sigs:
                n_duplicate += 1
                continue
            existing_sigs.add(c["selection_signature"])
            novel_children.append(c)

        population = sorted(population + novel_children,
                             key=lambda p: -p["fitness"])[:pop_size]

        # Sandbox-failure-rate diagnostic: does the model's OWN code run
        # cleanly, separate from whether it wins? A child can be valid
        # code that loses fairly (comprehension is fine, the idea is just
        # not competitive) or invalid/crashing code that loses by forfeit
        # (comprehension of the contract itself is the problem) — these
        # look identical in win_rate alone, so report this separately.
        llm_children = [c for c in children if not mock and c["used_llm"]]
        llm_failure_rate = (sum(c["n_sandbox_failures"] for c in llm_children) /
                             max(1, sum(c["n_steps"] for c in llm_children))
                             ) if llm_children else None

        history.append({"generation": gen,
                         "best_fitness": population[0]["fitness"],
                         "best_win_rate": population[0]["win_rate"],
                         "n_llm_failures_this_gen": sum(
                             (not c["used_llm"]) for c in children),
                         "llm_children_sandbox_failure_rate": llm_failure_rate,
                         "n_rank_equivalent_duplicates": n_duplicate})
        print(f"  gen {gen}: best fitness={population[0]['fitness']:.4f} "
              f"win_rate={population[0]['win_rate']:.3f} LOC={population[0]['loc']}"
              + (f"  llm_sandbox_failure_rate={llm_failure_rate:.3f}"
                 if llm_failure_rate is not None else "")
              + f"  duplicates={n_duplicate}/{len(children)}")

        # Stagnation-based early stop: patience generations with no
        # fitness improvement (float tolerance, same 1e-6 threshold
        # evaluate_af/_stagnant_prefix already use elsewhere in this file)
        # ends the run early instead of burning the remaining generations'
        # LLM-call budget on a population that has already converged.
        if population[0]["fitness"] > best_fitness_so_far + 1e-6:
            best_fitness_so_far = population[0]["fitness"]
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"  stopping early at gen {gen}: no fitness improvement "
                      f"for {patience} generations")
                break

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
    ap.add_argument("--train_dir", default=str(pathlib.Path(__file__).parent /
                                                "training_logs" / "train"))
    ap.add_argument("--pop_size", type=int, default=16)
    ap.add_argument("--n_generations", type=int, default=20)
    ap.add_argument("--n_offspring", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.005,
                     help="complexity penalty weight; fitness = win_rate - gamma*LOC")
    ap.add_argument("--mock", action="store_true", default=True)
    ap.add_argument("--real_llm", dest="mock", action="store_false")
    ap.add_argument("--model", default="qwen3-coder:30b",
                     help="Ollama model for --real_llm crossover/mutation. "
                          "qwen3-coder:30b (MoE, 19GB, fast per-call — good fit "
                          "for many small evolution-loop generations) is the "
                          "default; qwen3-coder-next (52-85GB) trades latency "
                          "for quality if evolved AFs look too shallow.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=6,
                     help="stop early if best fitness hasn't improved for this "
                          "many consecutive generations")
    ap.add_argument("--out_dir", default=str(pathlib.Path(__file__).parent /
                                              "evolution_runs" / "run1"))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_log_dir = out_dir / "af_code_logs"

    print(f"Loading training steps from {args.train_dir} ...")
    steps = load_training_steps(pathlib.Path(args.train_dir))
    print(f"Loaded {len(steps)} training steps.")
    if not steps:
        print("No training steps found — run generate_training_set.py first.")
        return

    result = run_evolution(
        steps, args.pop_size, args.n_generations, args.n_offspring, args.gamma,
        args.mock, args.model, args.seed, log_dir=code_log_dir,
        patience=args.patience,
    )

    best = result["population"][0]
    print(f"\nBest AF: fitness={best['fitness']:.4f} win_rate={best['win_rate']:.3f} "
          f"LOC={best['loc']}")
    print(f"\n{best['code']}")

    with open(out_dir / "best_af.py", "w") as f:
        f.write(best["code"])
    with open(out_dir / "history.json", "w") as f:
        json.dump({"history": result["history"], "mock": args.mock,
                    "n_llm_calls": result["n_llm_calls"],
                    "n_llm_failures": result["n_llm_failures"]}, f, indent=2)
    with open(out_dir / "final_population.json", "w") as f:
        json.dump([{k: v for k, v in p.items() if k != "term_weights"}
                    for p in result["population"]], f, indent=2)

    print(f"\nSaved best AF, history, and final population to {out_dir}")


if __name__ == "__main__":
    main()
