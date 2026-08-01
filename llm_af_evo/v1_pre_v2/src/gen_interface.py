"""
gen_interface.py — the candidate-GENERATION contract, a new evolution
surface distinct from af_interface.py's per-candidate SCORING contract.
Everything scored by score_pool() presupposes a pool already exists; this
contract is about how that pool gets proposed in the first place.

Motivated by the "AI-discovered tuning laws" paper (Tilbury et al. 2025,
bioRxiv 2025.11.12.688086): its cleanest transferable lesson is role
separation — one LLM call writes the equation form, a SEPARATE call writes
a geometric parameter initializer explicitly banned from calling solvers
(gradient descent handles refinement, not the initializer). The analog
here: a proposer that outputs candidate LOCATIONS, with no access to any
GP posterior, acquisition value, or gradient signal — geometry over
observed/Pareto structure only, exactly the EDGAR "no solver calls" ban
applied to this domain. This is deliberately weaker than an acquisition-
informed proposer (which could just badly reimplement qLogNEHVI); the
point of a geometry-only gate is a clean test of whether smart geometric
proposal ALONE can compete with the existing qLogNEHVI+UNSGA3 candidate
pool, before ever giving a proposer gradient/posterior access.

Contract (self-documenting-dict interface, matches af_interface.py's style)
-----------------------------------------------------------------------
Every candidate program is a Python source string defining exactly one
function:

    def propose_candidates(context) -> array-like, shape (n_candidates, n_dims)

context = {
    "X_obs": np.ndarray(n_obs, n_dims),      # every formulation observed so far, normalised to [0,1]^n_dims
    "pareto_x": np.ndarray(n_pf, n_dims),    # normalised x's of the CURRENT non-dominated observations
                                              # (n_pf may be 0 early in a campaign — handle that case)
    "campaign": {
        "step": int, "budget": int, "progress": float,   # step / budget
        "n_obs": int, "stagnant_batches": int,
    },
    "n_dims": int,
    "n_candidates": int,                     # how many candidate points to return
}

Returns: array-like of shape (n_candidates, n_dims), all entries in
[0, 1] (normalised bounds) — out-of-range values are clipped, non-finite
values or wrong shape are a sandbox rejection.

What's deliberately NOT in context, and why: no GP posterior, no
objective_names, no pareto_front (objective-space values), no
ref_point/ref_point_by_name, no acquisition value of any kind. A proposer
that wants to "look near the good stuff" only has pareto_x (geometric
location of non-dominated points in INPUT space) to work with — it cannot
know how good those points' outcomes were, only that they were
non-dominated. This is the whole point of the geometry-only gate: it
isolates "can smart proposal-of-locations alone add value" from "can
posterior-informed proposal add value" (the latter is a second, later
gate — see gen_sandbox.py's module docstring — and risks conflating any
win with 'reimplemented qLogNEHVI', not "LLM-discovered generation logic").
"""

import ast

import numpy as np

GEN_FUNCTION_NAME = "propose_candidates"

# numpy only, same restriction as af_interface.py's ALLOWED_GLOBALS — no
# scipy.stats.qmc (would hand the proposer a ready-made LHS/Sobol
# implementation instead of testing what an evolved geometric strategy
# looks like) and no sklearn (would hand it clustering for free).
ALLOWED_GLOBALS = {"np": np, "numpy": np}


def count_loc(code: str) -> int:
    """Same LOC-as-parsimony-proxy as af_interface.count_loc, retargeted
    at propose_candidates instead of score_pool."""
    try:
        tree = ast.parse(code)
        func = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == GEN_FUNCTION_NAME), None)
        if func is None:
            return len(code.strip().splitlines())
        start = func.body[0].lineno
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


# ── Seed proposers (hand-written, geometry-only gate) ──────────────────────
# Three structurally distinct geometric strategies, matching the composition
# pilot's discipline of testing multiple hand-designed seeds before any LLM
# evolution cost. All three are pure numpy, no GP/acquisition access.

SEED_LHS_UNEXPLORED = '''
def propose_candidates(context):
    """
    Coverage-maximising: draw a large random pool, greedily keep the
    n_candidates points that are furthest (max-min distance) from every
    already-observed point AND from each other. Pure exploration — ignores
    pareto_x entirely, doesn't care where good outcomes were, only cares
    about filling gaps in the observed input space.
    """
    d = context["n_dims"]
    n_cand = context["n_candidates"]
    X_obs = context["X_obs"]
    rng = np.random.default_rng(0)

    pool_size = max(n_cand * 20, 200)
    pool = rng.random((pool_size, d))

    chosen = []
    remaining = list(range(pool_size))
    for _ in range(n_cand):
        best_idx, best_dist = None, -1.0
        for i in remaining:
            ref_pts = X_obs if len(chosen) == 0 else np.vstack(
                [X_obs, pool[chosen]]) if len(X_obs) > 0 else pool[chosen]
            if len(ref_pts) == 0:
                dist = 1.0
            else:
                dist = float(np.linalg.norm(ref_pts - pool[i], axis=1).min())
            if dist > best_dist:
                best_idx, best_dist = i, dist
        chosen.append(best_idx)
        remaining.remove(best_idx)

    return pool[chosen]
'''

SEED_PERTURB_FRONT_EXTREMES = '''
def propose_candidates(context):
    """
    Local exploitation: perturb around the current Pareto-front x-locations
    with Gaussian noise, radius shrinking as the campaign progresses (wide
    early, tight late — the opposite of not knowing where to look). Falls
    back to uniform random if the front is empty (start of campaign).
    """
    d = context["n_dims"]
    n_cand = context["n_candidates"]
    pareto_x = context["pareto_x"]
    progress = context["campaign"]["progress"]
    rng = np.random.default_rng(1)

    if len(pareto_x) == 0:
        return rng.random((n_cand, d))

    radius = 0.25 * (1.0 - 0.7 * progress)
    idx = rng.integers(0, len(pareto_x), size=n_cand)
    base = pareto_x[idx]
    noise = rng.normal(0.0, radius, size=(n_cand, d))
    return np.clip(base + noise, 0.0, 1.0)
'''

SEED_HALF_EXPLOIT_HALF_EXPLORE = '''
def propose_candidates(context):
    """
    Splits the candidate batch: half perturbed near Pareto-front extremes
    (exploitation), half chosen by max-min distance from observed points
    (exploration) — the closest geometric analog to what a jointly-
    optimized batch acquisition does implicitly (balance both within one
    batch), but done by an explicit rule rather than joint MC integration.
    """
    d = context["n_dims"]
    n_cand = context["n_candidates"]
    X_obs = context["X_obs"]
    pareto_x = context["pareto_x"]
    progress = context["campaign"]["progress"]
    rng = np.random.default_rng(2)

    n_exploit = n_cand // 2
    n_explore = n_cand - n_exploit

    if len(pareto_x) == 0:
        exploit_pts = rng.random((n_exploit, d))
    else:
        radius = 0.25 * (1.0 - 0.7 * progress)
        idx = rng.integers(0, len(pareto_x), size=n_exploit)
        exploit_pts = np.clip(pareto_x[idx] + rng.normal(0.0, radius, size=(n_exploit, d)),
                               0.0, 1.0)

    pool_size = max(n_explore * 20, 100)
    pool = rng.random((pool_size, d))
    chosen = []
    remaining = list(range(pool_size))
    for _ in range(n_explore):
        best_idx, best_dist = None, -1.0
        for i in remaining:
            ref_pts = X_obs if len(chosen) == 0 else (
                np.vstack([X_obs, pool[chosen]]) if len(X_obs) > 0 else pool[chosen])
            dist = 1.0 if len(ref_pts) == 0 else float(
                np.linalg.norm(ref_pts - pool[i], axis=1).min())
            if dist > best_dist:
                best_idx, best_dist = i, dist
        chosen.append(best_idx)
        remaining.remove(best_idx)
    explore_pts = pool[chosen]

    return np.vstack([exploit_pts, explore_pts])
'''

GEN_SEED_PROGRAMS = {
    "gen_lhs_unexplored": SEED_LHS_UNEXPLORED.strip("\n"),
    "gen_perturb_front_extremes": SEED_PERTURB_FRONT_EXTREMES.strip("\n"),
    "gen_half_exploit_half_explore": SEED_HALF_EXPLOIT_HALF_EXPLORE.strip("\n"),
}
