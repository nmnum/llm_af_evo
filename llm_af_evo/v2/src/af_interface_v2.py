"""
af_interface_v2.py — variant of af_interface.py's AF contract for a smaller,
minimal-seed evolution design (see research_seed_population.md's
recommendation, written after reading FunBO/FunSearch/AlphaEvolve and the
Harris-group tuning-curve-equation-discovery paper): all four of those prior
systems seed with either one (FunBO/FunSearch/AlphaEvolve) or two near-
identical (Harris group) hand-written programs, relying on evolution itself
— not a large strategy-diverse hand-written seed set — to generate
diversity. af_interface.py's original 7-seed design was a deliberate
departure from that pattern; this module tries the pattern those systems
actually use instead, WITHOUT modifying af_interface.py/evolve_af.py or
re-running/invalidating any existing evolution_runs/ output that used them.

Design: exactly ONE full hand-written seed program is kept (trust_only) —
its only job is to anchor the exact contract syntax and conventions (dict
key names, the all-maximise mu/sigma convention af_interface.py's own
docstring warns is silently, catastrophically easy to get backwards) that a
bare one-line description can't reliably convey to an LLM. Every other
strategy this repo's original 7 seeds covered is instead described as a
one-line STRATEGY_HINT; evolve_af_v2.py's real-LLM mode asks the LLM to
WRITE score_pool implementing each hint at generation 0, rather than
shipping a hand-written implementation for it.

Mock mode (no LLM available in this sandbox) cannot turn a hint into code,
so it has nothing to do with STRATEGY_HINTS at all — see
evolve_af_v2.run_evolution's population-init logic, which uses only
SEED_PROGRAMS + random-term-weight padding in mock mode, same mechanism as
the original evolve_af.py just with fewer hand-written seeds.

Everything else (the contract itself, batch selection, LOC counting,
docstring extraction) is unchanged from af_interface.py and re-exported
here, not duplicated, so a fix to the shared contract logic never needs to
be made in two places.

---
Hint-set revision (v2.1): the original 6 hints were all UCB-style variants
(mu_sum +/- w*sigma_sum with different w(progress) schedules).  On the mAb
evolution run (run_v2_mAb_real_100), all 8 surviving AFs were simple UCB
variants — the LLM could not escape UCB space because the hint set offered
nothing else.  Three of the original 6 (novelty_only, ehvi_approx,
mc_hvi_approx) did not survive on mAb and are replaced here by upgraded
versions that use the same mechanism but with information the originals
lacked (Y_obs for outcome-space novelty, front resampling for noisy-front
HVI).  Three genuinely new mechanisms are added: DPP-based diversity
(Nava/Mutny/Krause 2021), greedy local penalization (Gonzalez et al. 2016),
and Pareto-membership trust-region approximation (inspired by NOSTRA,
Ghasemzadeh et al. 2025).  The result is 8 hints spanning 8 distinct
mechanisms, not 8 variants of UCB.

Prerequisite: hints noisy_front_hvi, outcome_novelty, and
pareto_membership require context["Y_obs"] (observed outputs) and
context["pareto_front"] (current non-dominated set) to be present in the
context dict.  pareto_front is already in context; Y_obs must be added by
the strategy code that builds context (one-line change, does not alter what
is evolved — score_pool still only ranks a fixed pool).
"""

import pathlib
import sys

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

from af_interface import (  # noqa: F401 — re-exported for evolve_af_v2.py
    AF_FUNCTION_NAME, ALLOWED_GLOBALS, OBJECTIVE_NAMES,
    select_batch, count_loc, extract_af_docstring,
    SEED_TRUST_ONLY,
)

# The one full hand-written seed kept in this design — pure exploitation,
# the simplest score_pool that still demonstrates the contract's exact
# conventions (iterating context["objective_names"] rather than hardcoding
# names, no re-flipping of the already-all-maximised mu/sigma values).
SEED_PROGRAMS = {
    "trust_only": SEED_TRUST_ONLY.strip("\n"),
}
SEED_TERM_WEIGHTS = {
    "trust_only": {"mu_sum": 1.0},
}

# Strategy hints: 8 distinct mechanisms spanning UCB, novelty, HV
# improvement, repulsive diversity, greedy penalization, and Pareto-
# membership trust regions.  evolve_af_v2.py's real-LLM mode asks the LLM
# to WRITE score_pool implementing each hint at generation 0.
#
# Hints requiring context["Y_obs"]: noisy_front_hvi, outcome_novelty
# Hints requiring context["pareto_front"]: noisy_front_hvi, pareto_membership
# Hints requiring neither: fixed_ucb, ucb_plus_novelty, phase_decaying_ucb,
#                          dpp_diversity, local_penalization
STRATEGY_HINTS = {
    # --- UCB family (kept from v2.0; survived on mAb) ---
    "fixed_ucb":
        "Score by predicted objective sum plus a fixed-weight (beta=2.0) "
        "uncertainty bonus, UCB-style — no novelty term, no phase-"
        "awareness.",
    "ucb_plus_novelty":
        "Combine a UCB-style exploration credit (predicted objective sum "
        "plus a fixed-weight uncertainty bonus) with an explicit novelty "
        "term (distance to the nearest observed point).",
    "phase_decaying_ucb":
        "Phase-aware: weight the uncertainty term heavily early in the "
        "campaign and let it decay as the budget is spent, exploiting more "
        "as the campaign progresses, with an extra novelty boost when the "
        "campaign has been stagnant (no hypervolume improvement for "
        "several batches).",

    # --- Noisy-front HV improvement (replaces ehvi_approx + mc_hvi_approx) ---
    # Requires: context["Y_obs"], context["ref_point"], context["pareto_front"]
    # v2.2: rewritten after a first attempt (see evolution_runs/
    # run_v2_coatings_gamma001) produced a broken non-dominated filter (a
    # hand-written double loop whose OR condition kept nearly every point,
    # not just the non-dominated ones) and an "improvement" formula that
    # degenerated into raw hypercube volume to ref_point regardless of the
    # front's position — i.e. the same absolute-HV-not-improvement mistake
    # mc_hvi_approx already failed on. This version spells out the exact
    # correct dominance test and warns against the ref_point-only shortcut.
    "noisy_front_hvi":
        "Treat the current Pareto front as an estimate, not ground truth: "
        "resample rows of context['Y_obs'] WITH REPLACEMENT (e.g. 10-20 "
        "bootstrap draws), and for each resample compute ITS OWN "
        "non-dominated subset using a CORRECT vectorized dominance test — "
        "for maximised objectives, point p is dominated by point q iff "
        "np.all(q >= p) and np.any(q > p); p is non-dominated iff no "
        "other point in the resample dominates it. Do not hand-roll this "
        "with nested Python loops and boolean OR chains — get the "
        "dominance direction exactly right (a point dominates another "
        "only if it is weakly better in every objective AND strictly "
        "better in at least one; do not confuse 'weakly dominates' with "
        "'does not dominate'). For each candidate, on each resampled "
        "front, compute its IMPROVEMENT: how much larger its own "
        "dominated hypervolume against context['ref_point'] is when "
        "ADDED to that resampled front, compared to the resampled front's "
        "hypervolume WITHOUT the candidate — not the candidate's raw "
        "volume-to-ref_point alone (a candidate deep inside a strong "
        "front should score near zero even if its raw distance to "
        "ref_point is large, because it adds nothing new). Average this "
        "improvement across the resampled fronts.",

    # --- Outcome-space novelty (replaces novelty_only; BEACON-style) ---
    # Requires: context["Y_obs"]
    "outcome_novelty":
        "Score candidates by how far their predicted outcomes are from "
        "anything already observed, not by how good those outcomes look: "
        "for each candidate, compare its gp_posterior mean vector to "
        "every row of context['Y_obs'] (the full observation history — "
        "not context['pareto_front'], which only holds the non-dominated "
        "subset and would understate novelty against dominated-but-"
        "nearby past observations) and favor candidates whose nearest-"
        "neighbor distance in objective space is large. This rewards "
        "exploring outcome regions the campaign hasn't tried yet, "
        "independent of predicted quality — most useful early, before "
        "the front is well resolved.",

    # --- DPP-based diversity re-ranking (new; Nava/Mutny/Krause 2021) ---
    # Requires: pool candidates' x and gp_posterior only
    # v2.2: rewritten after a first attempt computed `quality * diversity`
    # directly on raw predicted-objective sums. That inverts the intended
    # effect whenever quality is negative — which happens routinely here,
    # since min-direction objectives (e.g. coatings' conductance_std) are
    # sign-flipped to negative before the GP is fit, so a real, physically
    # good candidate can easily have a negative raw quality sum. This
    # version requires normalising quality to be non-negative first.
    "dpp_diversity":
        "Rank the pool for joint diversity, not just individual quality: "
        "build a similarity kernel between candidates (e.g. "
        "exp(-distance) based on distance in x or in predicted-objective "
        "space from gp_posterior, EXCLUDING each candidate's similarity "
        "to itself from any averaging) and prefer candidates that are "
        "both individually strong and mutually dissimilar from other "
        "high-scoring candidates, so that if two candidates are "
        "near-duplicates only one of them should end up highly ranked. "
        "IMPORTANT: before combining quality and diversity multiplicatively, "
        "normalise the raw quality (predicted objective sum) to be "
        "NON-NEGATIVE first, e.g. via min-max normalisation across the "
        "pool: (quality - min(qualities)) / (max(qualities) - "
        "min(qualities) + 1e-9). Multiplying by a diversity term "
        "in [0,1] when quality can be negative silently inverts the "
        "ranking for low-quality candidates (rewards them for being "
        "similar to others, not different) — do not skip the "
        "normalisation step. This does not need context['Y_obs'] or "
        "context['pareto_front'] — it operates purely on the pool's own "
        "predicted positions.",

    # --- Greedy local penalization (new; Gonzalez et al. 2016) ---
    # Requires: pool candidates' x and gp_posterior only
    # v2.3: rewritten AGAIN after a second attempt still inverted the
    # direction, despite v2.2 spelling out "near 0 for close, near 1 for
    # far" — because v2.2 called that quantity a "penalty factor", which
    # reads naturally as "amount to remove", and the model then applied
    # final_score = base_score * (1 - penalty_factor) as if penalty_factor
    # were a subtractive amount, silently re-inverting the direction back
    # to its original (wrong) sense. This version never uses the word
    # "penalty" for the thing that gets multiplied, states the exact
    # variable name to use, and explicitly forbids wrapping it in
    # (1 - x) anywhere afterward.
    "local_penalization":
        "Rank the pool greedily rather than scoring every candidate "
        "independently and in isolation: take the single best candidate "
        "by base score, then reduce the scores of remaining candidates "
        "in proportion to their CLOSENESS (in x or predicted-objective "
        "space) to candidates already picked so far, and repeat. Compute "
        "a variable called `multiplier` (not 'penalty', not 'suppression "
        "factor' — call it exactly `multiplier`) that you will multiply "
        "directly onto each remaining candidate's base score with NO "
        "further transformation: final_score = base_score * multiplier, "
        "full stop, never final_score = base_score * (1 - multiplier) or "
        "any other wrapping. `multiplier` itself must be LOW (near 0.0) "
        "when a candidate is CLOSE to an already-picked candidate, and "
        "HIGH (near 1.0) when a candidate is FAR from every already-"
        "picked candidate — e.g. `multiplier = 1.0 - exp(-dist)` is "
        "CORRECT (near 0 at dist=0, near 1 as dist grows) and you must "
        "use it exactly as-is, not re-inverted. Never let multiplier go "
        "negative. This keeps a tight cluster of near-identical "
        "high-scoring candidates from dominating the whole batch, "
        "without needing context['Y_obs'] at all — it only needs the "
        "pool's own positions.",

    # --- Pareto-membership trust region (new; NOSTRA-inspired) ---
    # Requires: context["pareto_front"], pool gp_posterior
    # v2.3: rewritten AGAIN after a second attempt fixed the performance
    # problem (single pass, no O(n^2) resampling) but got the dominance
    # DIRECTION backwards: it checked "does this sample dominate an
    # existing front point" (any(sample dominates pf_point for pf_point
    # in pf)) rather than "is this sample itself NOT dominated by the
    # front" (the actual definition of Pareto-optimality relative to a
    # front) — those are different, and measurably rarer/stricter,
    # events. A sample can be non-dominated by the front (genuinely
    # Pareto-optimal, e.g. a trade-off point) without dominating any
    # single existing front point outright. This version states the
    # correct direction as an explicit, unambiguous formula.
    "pareto_membership":
        "Estimate each candidate's probability of being Pareto-optimal "
        "by Monte Carlo sampling from its own gp_posterior (use a SMALL "
        "sample count, e.g. 20-30, not 100+). For each sample s (a "
        "vector of predicted objective values, all maximised), s counts "
        "as Pareto-optimal-so-far if and only if NO point q in "
        "context['pareto_front'] dominates s — where 'q dominates s' "
        "means ALL(q[i] >= s[i] for every objective i) AND ANY(q[i] > "
        "s[i] for some objective i). Do NOT check the reverse direction "
        "(whether s dominates some q) — that measures something "
        "different and rarer. A candidate's Pareto-probability is the "
        "fraction of its samples for which NO front point dominates it. "
        "Use context['pareto_front'] for this check, not context['Y_obs'] "
        "(which includes dominated points and would make every candidate "
        "look more Pareto-likely than it is). CRITICAL FOR PERFORMANCE: "
        "compute each candidate's own Pareto-probability EXACTLY ONCE, "
        "in a single pass over context['pool'], and store the results "
        "(e.g. in a list/array) — do NOT re-sample or re-score any "
        "candidate's posterior again inside a second nested loop over "
        "other candidates; reuse the already-computed probabilities. "
        "Then favor candidates that fall in the pool's densest region of "
        "high-Pareto-probability points (measured using only the "
        "ALREADY-COMPUTED probabilities and each candidate's x, e.g. "
        "inverse-distance-weighted density among candidates with "
        "probability above some threshold), rather than treating every "
        "high-scoring candidate as equally worth picking — approximating "
        "a trust region over the fixed pool instead of genuinely "
        "redirecting where the search samples next. Keep the total cost "
        "roughly O(n_candidates * n_samples), not "
        "O(n_candidates^2 * n_samples) — the sandbox has a 10-second "
        "timeout.",
}
