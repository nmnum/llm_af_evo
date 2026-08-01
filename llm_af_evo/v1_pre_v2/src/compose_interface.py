"""
compose_interface.py — the BATCH-COMPOSITION contract: a genuinely joint
alternative to af_interface.py's decompose-then-select pipeline
(score_pool -> select_batch). Motivated by the batch-size ablation ruling
out redundancy-discounting as the explanation for the mean-scorer tie
(the tie was flat across q=1..10, including q=1, where redundancy
accounting is structurally irrelevant) — compose_batch instead targets
MARGINAL HYPERVOLUME CONTRIBUTION: choosing the k-th batch member to
maximize the ADDITIONAL dominated volume given the k-1 members already
chosen, the way qEHVI's own greedy batch construction works. This is
genuinely joint outcome-space reasoning, never exercised by shrinking
batch size inside the old decomposed score->select pipeline (that pipeline
has no way to make candidate B's score depend on whether candidate A is
already in the batch).

Contract
--------
    def compose_batch(context, k) -> list[int]   # k indices into context["pool"]

context is IDENTICAL to af_interface.py's score_pool contract (same pool/
X_obs/objective_names/pareto_front/pareto_front_range/ref_point/
ref_point_by_name/campaign keys) — compose_batch sees the same GP-scored
candidate pool, just gets to reason about the whole batch jointly instead
of returning one independent score per candidate. Whether context["pool"]
entries' gp_posterior came from independent per-objective GPs or a
coregionalized DA-COREG multi-task GP is invisible to compose_batch —
that choice lives entirely in how the caller built pool_mu/pool_sigma
upstream (see da_coreg.py), keeping this contract orthogonal to the
surrogate-quality lever, per the 2x2 ablation design.

Returns: a list of exactly k DISTINCT integer indices into context["pool"],
0 <= index < len(context["pool"]).

Guard against the obvious gaming failure mode named in the plan
(repeating one high-mu index k times to trivially "diversify" nothing):
duplicate indices are a HARD REJECT at the sandbox validation layer, not
a soft penalty compose_batch has to reliably learn to avoid on its own —
foreclosing it structurally is simpler and safer than testing whether
it's exploitable under 2b replay.

Not carried over from af_interface.py: select_batch. Batch selection is
NOT a separate fixed step here — that's the entire point of this
contract; compose_batch IS the selection, expressed jointly instead of
score-then-top-k.
"""

import ast

import numpy as np

COMPOSE_FUNCTION_NAME = "compose_batch"

ALLOWED_GLOBALS = {"np": np, "numpy": np}


def count_loc(code: str) -> int:
    """Same LOC-as-parsimony-proxy as af_interface.count_loc/gen_interface.count_loc,
    retargeted at compose_batch."""
    try:
        tree = ast.parse(code)
        func = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == COMPOSE_FUNCTION_NAME), None)
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


# ── Seed compose_batch programs ────────────────────────────────────────────

SEED_GREEDY_MARGINAL_HV = '''
def compose_batch(context, k):
    """
    Greedy marginal-hypervolume batch construction — the direct joint-
    composition analog of qEHVI's own greedy batch strategy, done here
    over posterior MEANS only (no MC integration over uncertainty — that
    lever is what mc_hvi_approx tests separately, kept orthogonal here so
    a win is attributable to joint composition specifically). At each of
    k steps, picks the pool candidate whose posterior mean adds the most
    ADDITIONAL dominated hypervolume given the front UNION the batch
    members already chosen this call — a quantity that depends on what
    else is in the batch, which no independent per-candidate score can
    express by construction.
    """
    names = context["objective_names"]
    ref = context["ref_point_by_name"]
    ref_arr = np.array([ref[name] for name in names])
    front = context["pareto_front"]
    pool = context["pool"]
    n = len(pool)
    means = np.array([[cand["gp_posterior"][name]["mean"] for name in names]
                       for cand in pool])

    # ONE shared Monte-Carlo grid for every hv_of call in this invocation,
    # bounded by the union of the front and the whole candidate pool (not
    # recomputed per-call from whatever points happen to be passed) — this
    # matters because marginal gain is a DIFFERENCE of two hv_of calls
    # (with vs. without a candidate); if each call used its own bounding
    # box and its own fresh sample draw, the two estimates would live on
    # different grids and the difference would carry extra sampling noise
    # on top of the quantity actually being compared. Also turns ~125
    # fresh 800-sample draws per compose_batch call into exactly one.
    all_pts = (list(front) if len(front) > 0 else []) + list(means)
    hi_global = np.max(np.array(all_pts), axis=0) if len(all_pts) > 0 else ref_arr + 1.0
    span = np.maximum(hi_global - ref_arr, 1e-9)
    grid_n = 800
    rng = np.random.default_rng(0)
    samples = ref_arr + rng.random((grid_n, len(names))) * span
    grid_volume = float(np.prod(span))

    def hv_of(points):
        # Standard Monte-Carlo hypervolume estimate: fraction of samples,
        # drawn uniformly over [ref_arr, hi_global], dominated by ANY point
        # in `points` (i.e. componentwise <= that point — all samples are
        # already >= ref_arr by construction), times the sampled box's
        # volume. `dominated |=` unions across points, so this estimates
        # the volume of the UNION of each point's box, not a naive sum
        # that would double-count overlaps.
        if len(points) == 0:
            return 0.0
        dominated = np.zeros(grid_n, dtype=bool)
        for p in points:
            dominated |= np.all(samples <= p, axis=1)
        return float(np.mean(dominated) * grid_volume)

    accepted_pts = list(front) if len(front) > 0 else []
    chosen = []
    remaining = list(range(n))
    base_hv = hv_of(accepted_pts)
    for _ in range(min(k, n)):
        best_idx, best_gain = None, -1.0
        for i in remaining:
            trial_hv = hv_of(accepted_pts + [means[i]])
            gain = trial_hv - base_hv
            if gain > best_gain:
                best_idx, best_gain = i, gain
        chosen.append(best_idx)
        accepted_pts.append(means[best_idx])
        base_hv = hv_of(accepted_pts)
        remaining.remove(best_idx)

    return chosen
'''

SEED_GREEDY_MARGINAL_HV_MC = '''
def compose_batch(context, k):
    """
    Uncertainty-aware greedy marginal-hypervolume composition — combines
    the two mechanisms that were each null when tested alone:
    greedy_marginal_hv's joint batch composition (which LOST to baseline,
    apparently from overcommitting to unreliable posterior-MEAN estimates
    early in a campaign, with no uncertainty term to hedge against being
    wrong) and mc_hvi_approx's Monte-Carlo integration over posterior
    uncertainty (which tied baseline as an independent per-candidate
    score). Here, each candidate's marginal HV contribution is averaged
    over MC samples from its own posterior N(mean, std) — not a true
    joint sample across candidates (the numpy-only sandbox can't cheaply
    support a full cross-candidate covariance), just per-candidate
    marginal uncertainty — instead of using the raw posterior mean the
    way greedy_marginal_hv did. Once a candidate is accepted into the
    batch, its MEAN (not a sample) is what subsequent steps condition on,
    matching how a real campaign would only ever observe one true
    outcome per point, not a distribution.
    """
    names = context["objective_names"]
    ref = context["ref_point_by_name"]
    ref_arr = np.array([ref[name] for name in names])
    front = context["pareto_front"]
    pool = context["pool"]
    n = len(pool)
    means = np.array([[cand["gp_posterior"][name]["mean"] for name in names]
                       for cand in pool])
    stds = np.array([[max(cand["gp_posterior"][name]["std"], 1e-9) for name in names]
                      for cand in pool])

    n_mc = 6
    rng = np.random.default_rng(0)
    # Per-candidate MC samples, drawn ONCE up front (not re-drawn per
    # batch step) — same "fixed scenario set" reasoning as
    # greedy_marginal_hv's shared HV grid: comparing marginal gain across
    # different remaining candidates, and across successive k steps,
    # needs to be over consistent samples, not freshly redrawn noise each
    # time.
    mc_samples = np.stack([rng.normal(means[i], stds[i], size=(n_mc, len(names)))
                            for i in range(n)])  # (n, n_mc, M)

    all_pts = (list(front) if len(front) > 0 else []) + list(means)
    hi_global = np.max(np.array(all_pts), axis=0) if len(all_pts) > 0 else ref_arr + 1.0
    span = np.maximum(hi_global - ref_arr, 1e-9)
    grid_n = 500
    grid_rng = np.random.default_rng(1)
    hv_samples = ref_arr + grid_rng.random((grid_n, len(names))) * span
    grid_volume = float(np.prod(span))

    def dominated_mask(points):
        mask = np.zeros(grid_n, dtype=bool)
        for p in points:
            mask |= np.all(hv_samples <= p, axis=1)
        return mask

    accepted_pts = list(front) if len(front) > 0 else []
    chosen = []
    remaining = list(range(n))

    for _ in range(min(k, n)):
        dominated_accepted = dominated_mask(accepted_pts)
        base_hv = float(np.mean(dominated_accepted) * grid_volume)
        best_idx, best_gain = None, -1e18
        for i in remaining:
            # Vectorised across the n_mc samples for candidate i: for each
            # grid point, is it newly dominated by any of this
            # candidate's sampled outcomes, on top of what accepted_pts
            # already dominates?
            add_dominated = np.all(
                hv_samples[:, None, :] <= mc_samples[i][None, :, :], axis=2)  # (grid_n, n_mc)
            combined = dominated_accepted[:, None] | add_dominated       # (grid_n, n_mc)
            trial_hvs = combined.mean(axis=0) * grid_volume              # (n_mc,)
            gain = float(trial_hvs.mean() - base_hv)
            if gain > best_gain:
                best_idx, best_gain = i, gain
        chosen.append(best_idx)
        accepted_pts.append(means[best_idx])
        remaining.remove(best_idx)

    return chosen
'''

COMPOSE_SEED_PROGRAMS = {
    "greedy_marginal_hv": SEED_GREEDY_MARGINAL_HV.strip("\n"),
    "greedy_marginal_hv_mc": SEED_GREEDY_MARGINAL_HV_MC.strip("\n"),
}
