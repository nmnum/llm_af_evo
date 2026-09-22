"""
af_interface.py — the AF (acquisition function) contract that every
evolved program must implement, plus a small seed library of hand-written
reference AFs to bootstrap the programs database, and the FIXED batch-
selection logic that sits outside evolution (per the confirmed design:
evolution only touches per-candidate scoring, never batch selection).

Contract (self-documenting-dict interface)
--------------------------------------------
Every candidate program is a Python source string defining exactly one
function:

    def score_pool(context) -> list[float]  # or np.ndarray, len(context["pool"])

context = {
    "pool": [
        {"x": np.ndarray(16,),               # normalised candidate features
         "gp_posterior": {
             "Tm":        {"mean": float, "std": float},  # higher is better
             "kD":        {"mean": float, "std": float},  # higher is better
             "viscosity": {"mean": float, "std": float},  # PRE-FLIPPED: higher is better
         }},
        ... one entry per candidate in the pool ...
    ],
    "X_obs": np.ndarray(n_obs, 16),          # every formulation observed so far
    "Y_obs": np.ndarray(n_obs, 3),           # every OUTCOME observed so far, all-maximised,
                                              # rows match X_obs, columns match objective_names
                                              # (not filtered to the front — use this for
                                              # anything that needs the full observation
                                              # history, e.g. novelty distance or resampling
                                              # a noisy Pareto front; use pareto_front below
                                              # for the front itself)
    "objective_names": ["Tm", "kD", "viscosity"],   # column order for the array forms below
    "pareto_front": np.ndarray(n_pf, 3),     # non-dominated points, all-maximised
                                              # (columns match objective_names)
    "pareto_front_range": {"Tm": float, "kD": float, "viscosity": float},
                                              # per-objective observed range (max-min),
                                              # precomputed so sigma normalisation
                                              # doesn't require re-deriving it from
                                              # pareto_front by column position
    "ref_point": np.ndarray(3,),             # HV reference point, all-maximised,
                                              # same column order as objective_names
    "ref_point_by_name": {"Tm": float, "kD": float, "viscosity": float},
    "campaign": {
        "step": int, "budget": int, "progress": float,   # step / budget
        "n_obs": int, "stagnant_batches": int,
    },
}

Returns: a score per pool entry, higher = more preferred candidate; length
and order must match context["pool"]. No oracle ground truth is available
or needed — everything above is computable online during a real
(non-synthetic) campaign.

Why a nested, named dict instead of positional numpy arrays: the earlier
positional interface (pool_mu[:, 2] for viscosity, already sign-flipped,
remember which axis is which) is exactly the kind of thing a model gets
right most of the time and silently, catastrophically wrong sometimes —
producing code that's syntactically valid and doesn't crash, so the
sandbox's shape/finiteness checks can't catch it, but is semantically
backwards (e.g. rewarding high viscosity). Every quantity that previously
required remembering a column position is now reached by name. Both a
plain-array form (pareto_front, ref_point, with objective_names as the key
to their column order) and a fully-named form (pareto_front_range,
ref_point_by_name) are provided for the front/ref-point quantities, since
some strategies want array math (custom HV-style reasoning) and some want
simple named lookups (a sigma/range ratio) — both idioms stay self-
documented either way.

One deliberate deviation from calling score_pool once per candidate (which
would be maximally simple/self-documenting): candidates are batched into
context["pool"] and scored by a SINGLE call returning all N scores at
once, not N separate sandboxed calls. Subprocess launch overhead
(~50-100ms) times N candidates (~20-25) times every training step (360)
times every generation's children would make per-candidate sandboxing
prohibitively slow — this is a performance constraint, not a
self-documentation tradeoff, and the per-candidate dict structure inside
context["pool"] preserves the semantic clarity the user asked for while
keeping execution to one subprocess call per (AF, step) pair.

Batch selection (fixed, NOT evolved): top batch_size candidates by score.
This keeps the complexity penalty meaningful — an evolved program can only
earn a lower LOC by expressing scoring logic more simply, not by also
hiding complexity in a custom selection routine.
"""

import ast

import numpy as np

AF_FUNCTION_NAME = "score_pool"
OBJECTIVE_NAMES = ["Tm", "kD", "viscosity"]

# Names available inside the sandboxed exec namespace. numpy only — no
# scipy/sklearn/torch, keeping evolved programs restricted to closed-form
# scoring expressions, not full model-fitting routines (that's approach K's
# territory, not L1's).
ALLOWED_GLOBALS = {"np": np, "numpy": np}


def select_batch(scores, batch_size: int) -> list:
    """Fixed batch-selection logic: top batch_size candidates by score."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = min(batch_size, n)
    # argsort ascending, take the top k, then reverse for score-descending
    # order (order doesn't affect which points are picked, only reporting).
    return list(np.argsort(scores)[-k:][::-1])


def count_loc(code: str, af_function_name: str = AF_FUNCTION_NAME) -> int:
    """
    Non-blank, non-comment-only line count — the complexity-penalty term.
    Uses ast to strip the docstring (if any) before counting, so a program
    can't game the penalty by moving logic into a giant docstring, but
    inline comments on code lines still count (matching the Harris group's
    LOC-as-parsimony-proxy, not a stricter static-complexity metric).

    af_function_name: defaults to AF_FUNCTION_NAME ("score_pool") — every
    v1-v5 caller gets identical behaviour to before this parameter existed.
    v6's delta-seed contract passes "modifier" instead (see sandbox.py's
    af_function_name plumbing).
    """
    try:
        tree = ast.parse(code)
        func = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == af_function_name), None)
        if func is None:
            return len(code.strip().splitlines())
        start = func.body[0].lineno
        # Skip a leading docstring node
        if (isinstance(func.body[0], ast.Expr)
                and isinstance(func.body[0].value, ast.Constant)
                and isinstance(func.body[0].value.value, str)):
            start = func.body[1].lineno if len(func.body) > 1 else func.body[0].end_lineno + 1
        end = func.body[-1].end_lineno
        lines = code.splitlines()[start - 1:end]
    except Exception:
        lines = code.strip().splitlines()

    n = 0
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#"):
            n += 1
    return n


def has_return_statement(code: str, af_function_name: str = AF_FUNCTION_NAME) -> bool:
    """
    True iff score_pool's body contains at least one `return` anywhere
    (top level or nested in a loop/if/etc.) — a cheap pre-sandbox check for
    the repetition-loop failure mode where an LLM burns its whole token
    budget on repeated hedging comments and never reaches a return
    statement, so score_pool implicitly returns None every call. Doesn't
    guarantee the return is reachable or correct (that's the sandbox's
    job) — this only catches the specific "ran out of budget before
    writing any return at all" case cheaply, before ever spawning a
    subprocess.

    af_function_name: see count_loc's docstring — same default/override
    pattern, same v6 use (checks for "modifier" instead of "score_pool").
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    func = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == af_function_name), None)
    if func is None:
        return False
    return any(isinstance(n, ast.Return) for n in ast.walk(func))


def extract_af_docstring(code: str, af_function_name: str = AF_FUNCTION_NAME) -> str:
    """
    First line of score_pool's docstring, or "" if there is none/it's
    empty/the code doesn't parse. Shared by sandbox.py (docstring-presence
    enforcement) and evolve_af.py (results table, best-so-far prompt line)
    so both read the explainability line the same way. Only the first
    line is ever surfaced — a docstring may continue past it (existing
    seed AFs do), but everything past line 1 is prose for a human reader,
    not part of the enforced contract.

    af_function_name: see count_loc's docstring — same default/override
    pattern, same v6 use (extracts "modifier"'s docstring instead of
    "score_pool"'s).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    func = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == af_function_name), None)
    if func is None:
        return ""
    doc = ast.get_docstring(func)
    if not doc:
        return ""
    return doc.splitlines()[0].strip()


# ── Seed programs (hand-written, bootstrap the programs database) ─────────

SEED_TRUST_ONLY = '''
def score_pool(context):
    """
    Pure exploitation: rank by predicted objective sum only. Iterates over
    context["objective_names"] rather than hardcoding Tm/kD/viscosity, so
    this works on any oracle's objective set. No sign-flipping is applied
    here — gp[name]["mean"] is ALREADY in all-maximise convention by the
    time score_pool sees it (upstream strategy code does that conversion
    before building context), so re-applying objective_directions here
    would double-flip and invert behaviour on any min-objective.
    context["obj_correlation"] (per-candidate cross-objective posterior
    correlation, DA-COREG surrogate only — see cov_aligned_ei below) is
    available in context but deliberately unused here.
    """
    names = context["objective_names"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        scores.append(sum(gp[name]["mean"] for name in names))
    return scores
'''

SEED_FIXED_UCB = '''
def score_pool(context):
    """Fixed-weight UCB-style: predicted objective sum plus a beta*sigma bonus."""
    beta = 2.0
    rng = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"] / rng["Tm"] + gp["kD"]["std"] / rng["kD"]
                      + gp["viscosity"]["std"] / rng["viscosity"])
        scores.append(mu_sum + beta * sigma_norm)
    return scores
'''

SEED_NOVELTY_ONLY = '''
def score_pool(context):
    """Pure novelty: rank by distance to the nearest observed point."""
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        scores.append(float(dists.min()))
    return scores
'''

SEED_UCB_PLUS_NOVELTY = '''
def score_pool(context):
    """UCB-style exploration credit plus an explicit novelty term."""
    beta = 2.0
    w_nov = 0.5
    X_obs = context["X_obs"]
    rng = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"] / rng["Tm"] + gp["kD"]["std"] / rng["kD"]
                      + gp["viscosity"]["std"] / rng["viscosity"])
        novelty = float(np.linalg.norm(X_obs - cand["x"], axis=1).min())
        scores.append(mu_sum + beta * sigma_norm + w_nov * novelty)
    return scores
'''

SEED_PHASE_DECAYING_UCB = '''
def score_pool(context):
    """
    Phase-aware: explore early (high sigma weight when little of the
    budget is spent), exploit late, with an extra novelty kick if the
    campaign has been stagnant. Demonstrates the kind of strategy a purely
    per-candidate (x, mu, sigma) -> score function cannot express.
    """
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    beta = 3.0 * (1.0 - progress)
    stagnation_boost = 1.0 + 0.3 * min(stagnant, 5)
    X_obs = context["X_obs"]
    rng = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / rng[name] for name in names)
        novelty = float(np.linalg.norm(X_obs - cand["x"], axis=1).min())
        scores.append(mu_sum + beta * sigma_norm + stagnation_boost * 0.3 * novelty)
    return scores
'''

SEED_EHVI_APPROX = '''
def score_pool(context):
    """
    Approximate hypervolume-improvement: NOT exact EHVI (that needs Monte
    Carlo integration over the posterior, which the sandbox's numpy-only
    import whitelist doesn't support) — a cheap proxy using only the
    posterior mean. Candidates dominated by the current Pareto front get
    a heavily discounted score (they can still improve HV by filling in
    volume between front points, just less reliably than a non-dominated
    point does); non-dominated candidates score by their dominated volume
    relative to the reference point. This gives evolution a starting point
    that actually uses context["pareto_front"]/context["ref_point"],
    unlike the other seeds here. Iterates over context["objective_names"]
    (not hardcoded Tm/kD/viscosity) so this works on any oracle's
    objective set — front/ref/gp[name]["mean"] are all ALREADY in
    all-maximise convention, so no direction sign-flipping is needed here
    either, same reasoning as trust_only.
    """
    names = context["objective_names"]
    front = context["pareto_front"]
    ref = context["ref_point_by_name"]
    ref_arr = np.array([ref[name] for name in names])
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        y = np.array([gp[name]["mean"] for name in names])
        dominated = bool(np.any(np.all(front >= y, axis=1) & np.any(front > y, axis=1)))
        vol = float(np.prod(np.maximum(y - ref_arr, 0.0)))
        scores.append(vol if not dominated else 0.1 * vol)
    return scores
'''

SEED_MC_HVI_APPROX = '''
def score_pool(context):
    """
    Monte-Carlo hypervolume-improvement: draws n_samples per-objective
    samples from each candidate's GP posterior N(mean, std) and averages
    the per-sample dominated-volume-relative-to-ref-point across samples,
    instead of ehvi_approx's single point-estimate (mean-only) proxy. A
    direct, numpy-only decomposition of what qLogNEHVI's "noisy" Monte
    Carlo integration over the posterior does — per-candidate, with no
    joint reasoning across OTHER candidates in the same batch (that lever
    was tested separately by the batch-size ablation and ruled out as the
    explanation for the mean-scorer tie). Tests whether noise-aware
    scoring specifically, not batch composition, is what mean-only scorers
    (trust_only, ehvi_approx) are missing. Same all-maximise/no-re-flip
    convention as the other seeds — gp[name]["mean"/"std"] are already in
    that convention by the time score_pool sees them.
    """
    names = context["objective_names"]
    front = context["pareto_front"]
    ref = context["ref_point_by_name"]
    ref_arr = np.array([ref[name] for name in names])
    rng = np.random.default_rng(0)
    n_samples = 20
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        means = np.array([gp[name]["mean"] for name in names])
        stds = np.array([max(gp[name]["std"], 1e-9) for name in names])
        samples = rng.normal(means, stds, size=(n_samples, len(names)))
        vols = []
        for y in samples:
            if len(front) > 0:
                dominated = bool(np.any(np.all(front >= y, axis=1)
                                         & np.any(front > y, axis=1)))
            else:
                dominated = False
            vol = float(np.prod(np.maximum(y - ref_arr, 0.0)))
            vols.append(vol if not dominated else 0.1 * vol)
        scores.append(float(np.mean(vols)))
    return scores
'''

SEED_COV_ALIGNED_EI = '''
def score_pool(context):
    """
    Use cross-objective posterior correlation to weight candidates by
    expected joint improvement. Starting point only, not a prescribed
    implementation — trust_only's mu_sum, boosted for candidates whose
    objective pairs the DA-COREG surrogate currently believes move
    together (positive obj_correlation: a gain on one objective predicts
    a gain on the correlated one too, i.e. more of the mu_sum estimate is
    jointly, not independently, achievable) and discounted for pairs
    believed to trade off (negative correlation). context["obj_correlation"]
    is {} on the independent-GP path (ModelListGP's per-objective models
    never see each other), so this collapses to plain trust_only there —
    correct, since there is no cross-objective signal to use in that case.
    """
    names = context["objective_names"]
    corr = context["obj_correlation"]
    scores = []
    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        corr_bonus = 0.0
        n_pairs = 0
        for key, vals in corr.items():
            corr_bonus += vals[i]
            n_pairs += 1
        if n_pairs > 0:
            corr_bonus /= n_pairs
        scores.append(mu_sum * (1.0 + 0.25 * corr_bonus))
    return scores
'''

SEED_PROGRAMS = {
    "trust_only": SEED_TRUST_ONLY.strip("\n"),
    "fixed_ucb": SEED_FIXED_UCB.strip("\n"),
    "novelty_only": SEED_NOVELTY_ONLY.strip("\n"),
    "ucb_plus_novelty": SEED_UCB_PLUS_NOVELTY.strip("\n"),
    "phase_decaying_ucb": SEED_PHASE_DECAYING_UCB.strip("\n"),
    "ehvi_approx": SEED_EHVI_APPROX.strip("\n"),
    "mc_hvi_approx": SEED_MC_HVI_APPROX.strip("\n"),
    "cov_aligned_ei": SEED_COV_ALIGNED_EI.strip("\n"),
}

# term_weights equivalents of the hand-written seeds above, expressed over
# mock_mutator.TERM_LIBRARY. Needed so that in mock mode, a seed selected as
# a crossover parent contributes its ACTUAL structure — without this, mock
# mode's parent_a.get("term_weights") would fall back to a fresh random
# program for any seed parent, silently discarding the seed's structure and
# defeating the point of bootstrapping the population with them.
# phase_decaying_ucb and ehvi_approx have no term_weights equivalent (see
# mock_mutator.py's TERM_LIBRARY docstring — it's a purely-additive term
# grammar and can't express either's non-linear structure) — left absent,
# falls back to random_program() if selected as a mock crossover parent.
SEED_TERM_WEIGHTS = {
    "trust_only": {"mu_sum": 1.0},
    "fixed_ucb": {"mu_sum": 1.0, "sigma_sum_norm": 2.0},
    "novelty_only": {"novelty_min": 1.0},
    "ucb_plus_novelty": {"mu_sum": 1.0, "sigma_sum_norm": 2.0, "novelty_min": 0.5},
}
