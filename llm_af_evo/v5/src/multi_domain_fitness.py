"""
multi_domain_fitness.py — v5's multi-domain combination + gating logic,
kept as a standalone module deliberately separate from evolve_af_v5.py's
evolution loop, so this pure control-flow/arithmetic can be verified
against exact worked numbers before any real GP-campaign evaluation is
wired in (per the af-evolution branch design discussion this implements).

Design recap (see that conversation for the full reasoning):
  - Combination: mean - LAMBDA_STD_PENALTY * std of per-domain CI_lower_16
    margins, not worst-case/min. Min-of-noisy-estimates is a "loser's
    curse" — systematically pulled down by whichever domain got unlucky
    that generation, not necessarily whichever the formula is genuinely
    worst on. mean-lam*std gets real cross-domain-consistency pressure
    (blocks a v3-style "great on 2, bad on 1" formula) without that
    instability, at the same per-domain campaign budget.
  - Per-domain input MUST already be a noise-aware estimate (e.g.
    evaluate_af_2b's bootstrap ci_lower_16), not a raw mean — combining
    raw per-domain means would let mAb's ~50% CV alone dominate the std
    term, making every formula look "inconsistent" from domain noise, not
    genuine cross-domain behavioural difference.
  - Gating: sequential, cheapest/cleanest domain first (tunable), most
    expensive/noisiest last (mAb) — see evaluate_multi_domain's docstring.
    Deliberately NOT parallel — see this session's CPU-contention bug,
    which silently broke ~half of a real evolution run's LLM calls by
    competing for cores with background evaluation work.
"""

import numpy as np

LAMBDA_STD_PENALTY = 0.5

# Filters "actively harmful" only, not "not great" — a formula that's
# neutral on tunable (ci_lower_16 ~ 0) should still proceed to coatings,
# since it might be genuinely good on real domains despite being neutral
# on synthetic. Kept small and negative for exactly this reason.
GATE_THRESHOLD = -0.01

# v5 run1 finding (2026-09): with a single flat GATE_FAIL_FITNESS for
# every failure, generations 0-40 (41 of 150 — 27% of the whole run)
# were completely stuck: EVERY candidate collapsed to the identical
# fitness value regardless of whether it missed the tunable gate by a
# hair or catastrophically, and regardless of how many later domains it
# might have passed. That gives evolutionary selection nothing to climb
# toward "passing" while everything is currently failing — population[0]
# was provably frozen on one unchanged gate-failed candidate that whole
# stretch (history.json's best_domain_scores logged the exact same
# tunable score, unchanged to the float, for 41 straight generations),
# and the run only escaped once a candidate happened to clear the gate
# by luck rather than by any directed pressure.
#
# Fixed by making a gate failure's fitness a function of (a) how many
# domains it got through before failing — dominant term, so "passed
# tunable+coatings, failed mAb" always ranks above "failed tunable
# immediately" regardless of scores — and (b) the failing domain's own
# ci_lower_16 as a fine-grained tiebreaker WITHIN the same
# domains-passed count, so "missed tunable by 0.001" ranks above "missed
# tunable by 0.5". Both still always below GATE_FAIL_BASE + (a
# realistic maximum domains-passed bonus), so a fully-failing candidate
# can never accidentally outrank a fully-passing one — see the bound
# check in evaluate_multi_domain's docstring below.
GATE_FAIL_BASE = -1_000_000.0
# Must dominate any realistic per-domain ci_lower_16 magnitude (bounded
# well under +/-1 in practice — it's a relative-margin bootstrap
# estimate) so domains-passed-count is ALWAYS the primary sort key among
# failures, never overridden by the tiebreaker term.
DOMAIN_PASS_BONUS = 100.0


def combine_domain_scores(domain_scores: dict, lam: float = LAMBDA_STD_PENALTY) -> float:
    """mean - lam*std of domain_scores.values() (per-domain CI_lower_16
    margins, {domain_name: score}). Uses SAMPLE standard deviation
    (ddof=1, i.e. divides by n-1, not population std's n) — verified
    directly against the design conversation's worked calibration table:
    domain_scores={0.8, 0.8, 0.0} with lam=0.5 must produce fitness=0.30,
    which only sample std reproduces (population std gives 0.34). With
    only 3 domains this distinction is NOT a rounding nicety — numpy's
    OWN DEFAULT is population std (ddof=0), so calling plain .std() here
    would have silently produced numbers ~15% off from the calibration
    this function is supposed to implement.

    Requires at least 2 domain scores to compute a std (matches this
    project's 3-training-domain design — tunable/coatings/mAb); with
    exactly 1 score, std is undefined (ddof=1 divides by n-1=0) and this
    raises rather than silently returning the raw score as if consistency
    were free.
    """
    vals = np.array(list(domain_scores.values()), dtype=float)
    if len(vals) < 2:
        raise ValueError(
            f"combine_domain_scores needs >= 2 domain scores to compute a "
            f"sample std (got {len(vals)}) — a single-domain fitness isn't "
            f"what this function is for; use the score directly instead.")
    mean = vals.mean()
    std = vals.std(ddof=1)
    return mean - lam * std


# SE floor: a domain callable's se can legitimately be 0.0 (evaluate_af_2b
# returns 0.0 for n_campaigns<=1, where ddof=1 sample std is undefined) —
# without a floor, 1/se**2 divides by zero. Small enough to still assign
# an effectively enormous (but finite) weight to a genuinely near-zero-
# noise domain, rather than distorting real precision differences between
# the normal-noise domains.
SE_FLOOR = 1e-4


def combine_domain_scores_weighted(domain_scores: dict, domain_ses: dict,
                                    lam: float = LAMBDA_STD_PENALTY,
                                    se_floor: float = SE_FLOOR) -> float:
    """Inverse-variance-weighted analog of combine_domain_scores above —
    v5 run1 finding (2026-09): the unweighted mean/std treats every
    domain's ci_lower_16 as equally trustworthy, but tunable/coatings/mAb
    have wildly different intrinsic noise (rough CVs ~2%/~15%/~50%), so
    at matched campaign counts their SEs differ by an order of magnitude
    or more. The unweighted std term ends up measuring mostly whichever
    domain is noisiest, not genuine cross-domain behavioural
    inconsistency — the exact thing lam*std was supposed to penalise.
    This weights each domain by 1/se**2 (standard inverse-variance
    weighting — the precision-weighted combination that minimises the
    combined estimate's own variance), so a noisy domain's score
    contributes less to both the weighted mean AND the weighted std, and
    a domain's disagreement with the others only inflates the penalty in
    proportion to how PRECISELY it disagrees, not how loudly.

    domain_ses: {domain_name: standard_error}, SAME KEYS as domain_scores
    (evaluate_af_2b's "se_mean_margin" field — see its own docstring).
    Each se is floored at se_floor before use (see SE_FLOOR's comment).

    Uses the "reliability weights" unbiased weighted-variance formula
    (denominator sum(w) - sum(w**2)/sum(w), not just sum(w)) rather than
    the naive weighted-population-variance formula (denominator sum(w))
    DELIBERATELY: with EQUAL weights (all domains equally precise), this
    formula reduces EXACTLY to combine_domain_scores' plain sample
    variance (ddof=1) above — sum(w)=n, sum(w**2)=n, denominator=n-1 —
    so this is a strict generalisation of the already-calibrated
    unweighted function, not an unrelated alternative that happens to
    coincide in a special case. Verified directly (see this module's own
    test suite): combine_domain_scores_weighted with all-equal ses
    reproduces combine_domain_scores' output on the original worked
    calibration table to float precision.

    Falls back to the plain (unweighted) sample std if the weighted
    variance's denominator degenerates to <= 0 (e.g. one domain's weight
    so overwhelmingly dominates the others that sum(w**2)/sum(w) >=
    sum(w) — only possible with extremely lopsided precision, not a
    normal case here) rather than dividing by zero or a negative number.
    """
    keys = list(domain_scores.keys())
    vals = np.array([domain_scores[k] for k in keys], dtype=float)
    if len(vals) < 2:
        raise ValueError(
            f"combine_domain_scores_weighted needs >= 2 domain scores to "
            f"compute a variance (got {len(vals)}) — a single-domain "
            f"fitness isn't what this function is for; use the score "
            f"directly instead.")
    ses = np.array([max(domain_ses[k], se_floor) for k in keys], dtype=float)
    weights = 1.0 / (ses ** 2)
    sum_w = weights.sum()
    sum_w2 = (weights ** 2).sum()
    weighted_mean = float(np.sum(weights * vals) / sum_w)
    denom = sum_w - sum_w2 / sum_w
    if denom <= 0:
        weighted_std = float(vals.std(ddof=1))
    else:
        weighted_var = float(np.sum(weights * (vals - weighted_mean) ** 2) / denom)
        weighted_std = weighted_var ** 0.5
    return weighted_mean - lam * weighted_std


def evaluate_multi_domain(evaluate_fns: dict, lam: float = LAMBDA_STD_PENALTY,
                           gate_threshold: float = GATE_THRESHOLD,
                           gate_fail_base: float = GATE_FAIL_BASE,
                           domain_pass_bonus: float = DOMAIN_PASS_BONUS) -> tuple:
    """Sequential-gated multi-domain evaluation. evaluate_fns is an ORDERED
    dict {domain_name: zero-arg callable() -> (ci_lower_16, se) pair} —
    order is the gate sequence (e.g. tunable -> coatings -> mAb, cheapest/
    cleanest first, most expensive/noisiest last), and each callable is
    invoked LAZILY, in order, only as far as needed: as soon as one
    domain's score falls at or below gate_threshold, evaluation stops
    immediately — later domains' callables are never called at all. This
    is the actual cost saving of the funnel (see the design conversation's
    ~38-vs-96-campaigns/generation estimate): most candidates fail the
    cheap tunable gate and never touch mAb's expensive n_fitness_seeds
    reruns.

    The se half of each pair (evaluate_af_2b's "se_mean_margin") is used
    ONLY for the final combination among domains that all passed —
    combine_domain_scores_weighted's inverse-variance weighting (see its
    own docstring for why the plain unweighted mean/std was misleading
    once tunable/coatings/mAb's very different intrinsic noise levels are
    accounted for). Gating itself is unaffected: a domain's ci_lower_16
    is compared to gate_threshold directly regardless of its precision —
    a noisy domain that happens to score badly still fails its gate.

    A domain callable that raises is treated as that domain scoring
    "actively harmful" (gate_threshold - 1.0, guaranteed to fail the
    gate, se=0.0 — never used, since a gate failure doesn't reach the
    weighted-combination step) rather than propagating — a candidate that
    crashes on one domain's shape (e.g. assumes 2 objectives, crashes on
    a 3-objective domain) is genuinely harmful there, not merely
    unmeasured, and must not abort the whole multi-domain computation for
    that child.

    Returns (fitness, domain_scores, gate_failed_at):
      fitness: for a candidate that fails a gate, GATE_FAIL_BASE +
        (number of domains passed before failing) * DOMAIN_PASS_BONUS +
        (the failing domain's own score) — see GATE_FAIL_BASE's module
        comment for why this graded formula replaced a single flat
        constant (v5 run1 spent 41 of 150 generations completely stuck
        because every failure looked identical to selection). For a
        candidate that passes every gate: combine_domain_scores_weighted(
        domain_scores, domain_ses, lam). A failing candidate can NEVER
        outrank a passing one: even passing every domain but the last
        still adds at most (len(evaluate_fns) - 1) * domain_pass_bonus,
        which for any realistic domain count stays far below
        GATE_FAIL_BASE's magnitude — e.g. 3 domains, 2 passed: -1,000,000
        + 200 + score ~= -999,800, vs. a passing candidate's combined
        output, which is a small fraction (typically well under +/-1).
      domain_scores: {domain_name: ci_lower_16} for every domain ACTUALLY
        evaluated — a partial dict (fewer than len(evaluate_fns) entries)
        if gated out early, since later domains are never called. (The
        se half of each pair isn't returned here — callers that need it,
        e.g. for logging, should read it from wherever they built
        evaluate_fns; this return value stays the same shape it always
        was, for backward compatibility with checkpoint/reporting code
        that already expects {domain_name: float}.)
      gate_failed_at: the domain name that failed the gate, or None if
        every domain passed.
    """
    domain_scores = {}
    domain_ses = {}
    n_passed = 0
    for name, fn in evaluate_fns.items():
        try:
            score, se = fn()
        except Exception:
            score, se = gate_threshold - 1.0, 0.0
        domain_scores[name] = score
        domain_ses[name] = se
        if score <= gate_threshold:
            fail_fitness = gate_fail_base + n_passed * domain_pass_bonus + score
            return fail_fitness, domain_scores, name
        n_passed += 1
    fitness = combine_domain_scores_weighted(domain_scores, domain_ses, lam=lam)
    return fitness, domain_scores, None
