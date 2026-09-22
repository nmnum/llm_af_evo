"""
multi_domain_fitness_v6.py — v6's combination + gating logic. Standalone
and pure-arithmetic, same discipline as v5's multi_domain_fitness.py: this
module gets verified against hand-worked numbers before any real
per-domain evaluation is wired in.

Design recap (see v6/README.md for the full reasoning; this module
implements decisions 3 and part of the gate change there):

  - v5's `mean - lam*std` combination was found to have two compounding
    flaws, both confirmed against v5 run2's actual data:
      (a) the std penalty DUPLICATES the gate's job, in the wrong
          direction — the gate already enforces "not harmful anywhere"
          before combination runs at all, so a symmetric std penalty on
          top of that punishes UPSIDE variance (real wins on some
          domains) among already-safe candidates, not hidden badness.
      (b) raw-margin std conflates "large intrinsic domain volatility"
          (coatings/mAb naturally swing further than tunable even with
          infinite campaigns) with "genuine cross-domain inconsistency"
          — v5's inverse-variance weighting fixed the MEASUREMENT-noise
          version of this (how uncertain is each domain's estimate) but
          not the SCALE version (how big are real effects on this
          domain at the population level).
  - v6 fixes both by combining Z-SCORES (score/se), mean only, no std
    term at all:
      fitness = mean(z_1, ..., z_k) - gamma * loc
    Z-scoring is unit-free and directly comparable across domains
    regardless of native noise/CV, which addresses (b) more directly
    than weighting raw values; dropping the std term removes (a)
    entirely — cross-domain safety is the gate's job, not this
    function's.
  - Gate threshold also switches to z-units for the same reason: v5's
    fixed raw-margin epsilon (-0.01) meant very different things across
    domains at very different noise scales. v6 gates on
    `z <= GATE_Z_THRESHOLD` (proposed -1.0, i.e. "at least 1 SE below
    baseline" — comparable in spirit to the diagnostic methodology that
    motivated this whole revision: comparing observed deltas to their
    own SE before calling them real).
  - The graded gate-fail tiebreaker (v5's fix for the 41-generation
    flat-fitness stall) is KEPT, just re-expressed in z-units: a failing
    candidate's fitness is GATE_FAIL_BASE + (domains passed) *
    DOMAIN_PASS_BONUS + (failing domain's own z-score) — same ordering
    guarantees as v5's version (domains-passed dominates; z-score is a
    fine-grained tiebreaker within the same passed-count; a failing
    candidate can never outrank a passing one).
"""

import numpy as np

# "At least 1 SE below baseline" — comparable across domains regardless
# of native noise scale, unlike v5's fixed raw-margin epsilon.
GATE_Z_THRESHOLD = -1.0

# SE floor: a domain's measured se can legitimately be ~0 (e.g. very
# large n on a low-noise synthetic) — without a floor, score/se blows up
# to +/-inf for a borderline-zero score, dominating the mean for no
# genuine reason. Same value/rationale as v5's SE_FLOOR.
SE_FLOOR = 1e-4

# Same magnitudes as v5's GATE_FAIL_BASE/DOMAIN_PASS_BONUS, same
# rationale (see module docstring) — GATE_FAIL_BASE must dominate any
# realistic z-score magnitude (bounded well under +/-100 in practice for
# real domain SEs) so domains-passed-count is always the primary sort
# key among failures.
GATE_FAIL_BASE = -1_000_000.0
DOMAIN_PASS_BONUS = 100.0


def z_score(score: float, se: float, se_floor: float = SE_FLOOR) -> float:
    """score / max(se, se_floor) — the one place SE-flooring happens, so
    every caller in this module goes through the same floor."""
    return score / max(se, se_floor)


def combine_domain_z_scores(domain_scores: dict, domain_ses: dict,
                             se_floor: float = SE_FLOOR) -> float:
    """Mean of per-domain z-scores (score/se), NO spread/std term — see
    module docstring for why the std term was dropped entirely rather
    than reweighted again. domain_scores/domain_ses: same-keyed dicts
    ({domain_name: float}), same shape v5's combine_domain_scores_*
    functions used.

    Requires at least 1 domain (unlike v5's >= 2 requirement — there is
    no variance/std being computed here, so a single domain is a
    perfectly meaningful mean-of-one; the >= 2 requirement in v5 was
    specifically about std needing at least 2 points, which no longer
    applies).
    """
    keys = list(domain_scores.keys())
    if not keys:
        raise ValueError("combine_domain_z_scores needs >= 1 domain score")
    zs = [z_score(domain_scores[k], domain_ses[k], se_floor) for k in keys]
    return float(np.mean(zs))


def evaluate_multi_domain_v6(evaluate_fns: dict,
                              gate_z_threshold: float = GATE_Z_THRESHOLD,
                              gate_fail_base: float = GATE_FAIL_BASE,
                              domain_pass_bonus: float = DOMAIN_PASS_BONUS,
                              se_floor: float = SE_FLOOR) -> tuple:
    """Sequential-gated multi-domain evaluation, v6 version — same
    lazy-evaluation/short-circuit structure as v5's evaluate_multi_domain
    (see its own docstring for the funnel-cost-saving rationale, which is
    unchanged here), but gates and combines in Z-UNITS throughout instead
    of raw margins.

    evaluate_fns: ORDERED dict {domain_name: zero-arg callable() ->
    (score, se) pair} — order is the gate sequence (v6: ascending noise,
    e.g. ZDT1@15% -> DTLZ2-3@30% -> ZDT3@50%, cheapest/cleanest-read
    first, same funnel-efficiency reasoning as v5).

    A domain callable that raises is treated as "actively harmful"
    (z = gate_z_threshold - 1.0, guaranteed to fail the gate) — same
    crash-handling policy as v5's evaluate_multi_domain.

    Returns (fitness, domain_scores, domain_zs, gate_failed_at):
      fitness: for a gate failure, gate_fail_base + (domains passed) *
        domain_pass_bonus + (failing domain's own z-score) — see module
        docstring. For a candidate passing every gate:
        combine_domain_z_scores(domain_scores, domain_ses).
      domain_scores: {domain_name: raw score} for every domain actually
        evaluated (partial dict if gated out early).
      domain_zs: {domain_name: z-score} for every domain actually
        evaluated — new relative to v5's return shape (v5 only returned
        raw scores; v6 callers/loggers want the z-scores directly rather
        than recomputing them from scores+ses every time).
      gate_failed_at: the domain name that failed the gate, or None if
        every domain passed.
    """
    domain_scores = {}
    domain_ses = {}
    domain_zs = {}
    n_passed = 0
    for name, fn in evaluate_fns.items():
        try:
            score, se = fn()
        except Exception:
            score, se = 0.0, se_floor
            z = gate_z_threshold - 1.0
        else:
            z = z_score(score, se, se_floor)
        domain_scores[name] = score
        domain_ses[name] = se
        domain_zs[name] = z
        if z <= gate_z_threshold:
            fail_fitness = gate_fail_base + n_passed * domain_pass_bonus + z
            return fail_fitness, domain_scores, domain_zs, name
        n_passed += 1
    fitness = combine_domain_z_scores(domain_scores, domain_ses, se_floor)
    return fitness, domain_scores, domain_zs, None
