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
#
# egbo_novelty_like: v4 addition, added after direct measurement showed
# every LLM-invented "posterior resampling"/"Pareto probability" hand-
# approximation of hypervolume improvement (evolve_af_v4.py's
# MECHANISM_FAMILIES — dominated both v3 evolution runs' stagnation
# phases, never beat the population's real champion) is a strictly lower-
# fidelity approximation of a value now handed to score_pool directly:
# context["pool"][i]["acq_value_norm"] is botorch's own qLogNEHVI
# acquisition value (see sandbox.py's pool_acq_value docs / full_replay.py's
# strategy_evolved_af) — the EXACT real signal strategy_mo_egbo_novelty
# itself scores candidates with, MC-integrated over the full joint
# objective posterior, not reconstructed from per-objective mean/std. This
# seed combines it with novelty exactly the way EGBO's own
# novelty_aware_select_vectorised does (real w_acq=0.9/w_nov=0.1 defaults,
# both terms min-max normalised over the pool) — the closest static
# approximation of EGBO's actual scoring achievable in the score_pool
# contract; it only omits EGBO's SEQUENTIAL within-batch novelty term
# (distance to candidates already picked earlier in the SAME batch),
# which a one-shot, order-independent per-candidate score list cannot
# express (see run_2b_diagnostic_v4.py for a direct comparison against
# the real baseline).
SEED_EGBO_NOVELTY_LIKE = '''def score_pool(context):
    """EGBO-novelty-style: 0.9*acq_value_norm (real qLogNEHVI acquisition
    value, min-max normalised over the pool — the same signal EGBO's own
    baseline scores with) plus 0.1*novelty-to-nearest-observed-point (also
    min-max normalised), matching strategy_mo_egbo_novelty's real w_acq/
    w_nov defaults exactly. Omits only EGBO's sequential within-batch
    novelty term, which a static per-candidate score cannot express."""
    X_obs = context["X_obs"]
    dists = np.array([
        np.linalg.norm(X_obs - cand["x"], axis=1).min()
        for cand in context["pool"]
    ])
    d_min, d_max = dists.min(), dists.max()
    if d_max - d_min > 1e-12:
        nov_norm = (dists - d_min) / (d_max - d_min)
    else:
        nov_norm = np.full(len(dists), 0.5)
    acq_norm = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    return list(0.9 * acq_norm + 0.1 * nov_norm)
'''

SEED_PROGRAMS = {
    "trust_only": SEED_TRUST_ONLY.strip("\n"),
    "egbo_novelty_like": SEED_EGBO_NOVELTY_LIKE.strip("\n"),
}
SEED_TERM_WEIGHTS = {
    "trust_only": {"mu_sum": 1.0},
    # No mock_mutator.py term-weight representation exists for
    # acq_value_norm (mock mode's crossover only knows the fixed
    # TERM_TEMPLATES vocabulary) — .get() returns None for this seed in
    # run_evolution, which make_child already handles the same way it
    # handles every hint-generated seed (falls back to a fresh random
    # program if this ever becomes a mock-mode crossover parent). Real-LLM
    # mode's crossover works from the raw code string regardless, so this
    # only matters for mock-mode smoke tests, never a real run.
}

# Strategy hints: 10 distinct mechanisms spanning novelty, HV improvement,
# repulsive diversity, greedy penalization, Pareto-membership trust
# regions, front coverage, cross-objective correlation, and temporal
# momentum.  evolve_af_v2.py's real-LLM mode asks the LLM to WRITE
# score_pool implementing each hint at generation 0.
#
# v4.1: dropped fixed_ucb/ucb_plus_novelty/phase_decaying_ucb (the three
# pure-UCB-family hints kept from v2.0) and replaced them with
# front_coverage_gap/obj_correlation_bonus/improvement_momentum. Found
# necessary from evolve_af_v4.py run1's live behaviour (gens 22-45): once
# the champion converges to an acq_value_norm+uncertainty-bonus formula
# (which every run so far has — acq_value_progress_blend IS this family
# by design), these three hints are textually different but MECHANICALLY
# THE SAME FAMILY as the champion, so handing them to the model as a
# "structurally different" escape option during stagnation doesn't
# actually offer an escape — see evolve_af_v4.py's STRATEGY_HINT_TAGS/
# _UCB_UNCERTAINTY_TAG history for the full account (the model was handed
# ucb_plus_novelty as a "fresh" redirect while already stuck restating its
# own UCB+novelty formula). This isn't a claim the UCB family performs
# badly — docs/llm_evolved_afs_comprehensive_log.md §25's DTLZ2 table has
# hint_fixed_ucb (mean HV 4.745) beating every front-range-normalised
# variant tested — only that it's the wrong tool for a catalog whose job
# is to offer the model somewhere ELSE to go once it's already fixated on
# that exact family. The three replacements were checked against that same
# log before being written: none use front-range normalisation (closed as
# a mechanism there — §22-25, every domain and variant tested failed to
# beat hint_fixed_ucb) and none require a hand-rolled dominance/HV test
# (already the documented failure mode of noisy_front_hvi/
# pareto_membership/dro_robust_hvi below — the LLM gets these wrong
# repeatedly per this file's own hint-rewrite history).
#
# Hints requiring context["Y_obs"]: noisy_front_hvi, outcome_novelty,
#                                    improvement_momentum
# Hints requiring context["pareto_front"]: noisy_front_hvi, pareto_membership,
#                                           front_coverage_gap
# Hints requiring context["obj_correlation"]: obj_correlation_bonus (falls
#   back to a no-op when empty — see its own text; empty is the common case)
# Hints requiring neither: dpp_diversity, local_penalization
STRATEGY_HINTS = {
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

    # --- DRO-inspired robust MC-HVI (new; penalizes fragile HVI, not just mean) ---
    # Requires: context["pareto_front"], pool gp_posterior
    # Backtested by hand on 10 real mAb training campaigns before adding
    # this hint (no LLM calls): mean_margin +10.1% vs fixed_ucb's +7.2% and
    # noisy_front_hvi's +7.4%, mean_hv highest of the three (5931 vs 5812/
    # 5899) — encouraging but NOT conclusive at n=10 (mAb's fitness signal
    # is noise-dominated even at n=75, see evaluate_af_2b's GAMMA
    # CALIBRATION docstring and the --oracle excipient startup warning).
    "dro_robust_hvi":
        "Score candidates by a ROBUST estimate of hypervolume improvement, "
        "not just its average: for each candidate, draw n_samples (e.g. "
        "20) samples from its own gp_posterior N(mean, std) per objective "
        "using np.random.normal or a numpy Generator's .normal (all-"
        "maximise convention, no sign-flipping). For EACH sample s, "
        "compute a per-sample improvement value: if context['pareto_front'] "
        "is non-empty, check whether s is dominated by any front point "
        "(dominated iff ANY q in pareto_front has ALL(q >= s) AND ANY(q > "
        "s) — get this direction exactly right, same test as "
        "pareto_membership's hint above) and heavily discount s's raw "
        "volume-to-ref_point (np.prod(np.maximum(s - ref_point, 0))) when "
        "dominated (e.g. multiply by 0.1), full value when not dominated. "
        "This gives you n_samples improvement values per candidate — "
        "compute their MEAN and their STD. The final score MUST be "
        "`mean(improvement_values) - lam * std(improvement_values)` with a "
        "PLUS-becomes-MINUS penalty (lam is a positive weight, e.g. 1.0, "
        "left for you to tune) — candidates whose improvement swings a "
        "lot across samples (fragile under posterior noise, likely an "
        "extrapolated or noise-driven prediction) must score LOWER than "
        "equally-averaging candidates whose improvement is stable across "
        "samples. Do NOT write `mean + lam*std` (that is plain UCB-style "
        "optimism, not this hint) and do NOT wrap the std term in `(1 - "
        "x)` or any other transform — subtract it directly, exactly once. "
        "This does not need context['Y_obs'] — only context['pareto_front'] "
        "and each candidate's own gp_posterior.",

    # --- Real acquisition-value exploitation (v4 addition) ---
    # Requires: context["pool"][i]["acq_value_norm"] (v4-only context key
    # — see sandbox.py's pool_acq_value docs). Deliberately steers AWAY
    # from noisy_front_hvi/pareto_membership/dro_robust_hvi's own
    # hand-rolled hypervolume-improvement estimates: acq_value_norm IS
    # that exact quantity already, computed properly by botorch, so
    # re-deriving it by hand is now a strictly worse approximation of the
    # same thing, not a genuinely different idea.
    "acq_value_progress_blend":
        "Use context['pool'][i]['acq_value_norm'] — botorch's own "
        "qLogNEHVI acquisition value for this candidate, already min-max "
        "normalised into [0,1] across the pool. This is the SAME real "
        "signal EGBO's own baseline scores candidates with, not a "
        "hand-derived approximation — do NOT recompute your own "
        "hypervolume-improvement estimate from gp_posterior mean/std "
        "(e.g. via posterior resampling or dominance-probability "
        "sampling); acq_value_norm already IS that estimate, done "
        "properly via botorch's Monte Carlo integration over the FULL "
        "joint objective posterior, so re-deriving it by hand would be a "
        "strictly worse approximation of the exact same quantity. Use "
        "acq_value_norm as the PRIMARY/dominant term in your score, "
        "blended with a SMALLER secondary adjustment of your choosing — "
        "e.g. a campaign-progress-aware weight shift, a novelty term "
        "(distance to nearest observed point), or an uncertainty bonus — "
        "so the final score is not just acq_value_norm alone but a "
        "genuine blend where acq_value_norm clearly dominates.",

    # --- Front coverage gap (v4.1 addition, replaces fixed_ucb) ---
    # Requires: context["pareto_front"] (falls back to context["Y_obs"] if
    # the front has fewer than 3 points — very early in a campaign).
    "front_coverage_gap":
        "Reward candidates predicted to land in an UNDER-COVERED region of "
        "the current Pareto front, not just candidates that are near it: "
        "for each candidate, take its predicted objective vector "
        "(gp_posterior means, all-maximise convention) and compute its "
        "distance to the k=3 NEAREST points already in "
        "context['pareto_front'] (if context['pareto_front'] has fewer "
        "than 3 points, use context['Y_obs'] instead — early in a "
        "campaign the front itself is too small to give a meaningful "
        "density estimate). Take the MEAN of those k nearest-distances as "
        "the candidate's coverage-gap score: a candidate near a SPARSE "
        "stretch of the front scores high, a candidate near a DENSE "
        "cluster of front points scores low, even when both are equally "
        "close to the front overall. This is DIFFERENT from a plain "
        "novelty term (nearest-distance to ANY observed point, which "
        "rewards being far from everything indiscriminately) and from a "
        "pool-internal diversity term (pairwise spread WITHIN the "
        "candidate batch, not against the front): this hint specifically "
        "targets gaps in the FRONT's own coverage, so a candidate can "
        "score high here even while sitting close to several already-"
        "observed but dominated (off-front) points, as long as the front "
        "itself is sparse nearby. Blend as a SMALLER secondary term added "
        "to context['pool'][i]['acq_value_norm'] (see "
        "acq_value_progress_blend's guidance above), not as the sole "
        "score. Cost is O(n_candidates * n_front_points) — cheap at "
        "typical front sizes, well under the sandbox's 10-second budget.",

    # --- Cross-objective correlation bonus (v4.1 addition, replaces
    #     ucb_plus_novelty) ---
    # Requires: context["obj_correlation"] (optional, {} for most oracles
    # — see sandbox.py's own docstring for this key). MUST degrade to a
    # true no-op when empty; do not treat a missing key as zero-
    # correlation-is-itself-informative.
    "obj_correlation_bonus":
        "Use context['obj_correlation'] — an OPTIONAL, per-candidate "
        "cross-objective posterior correlation signal, only populated "
        "when the surrogate is a DA-COREG GP (empty dict {} for every "
        "other oracle/surrogate, which is the common case). When "
        "non-empty, it is keyed 'name_a,name_b' -> list[float] "
        "(index-aligned with context['pool'], one value per candidate) "
        "since JSON has no tuple keys. The idea: a candidate predicted to "
        "improve an objective that this signal says is NEGATIVELY "
        "correlated with objectives the current front already covers "
        "well has a marginal hypervolume contribution that tends to be "
        "disproportionately large, since progress there is not "
        "'automatically' bought by progress the campaign has already "
        "made on the correlated objectives. Build a small bonus from "
        "this (e.g. average the relevant per-candidate correlation "
        "values, weighting negative correlations positively) and blend "
        "it as a SMALLER secondary term on top of "
        "context['pool'][i]['acq_value_norm'], which stays the dominant "
        "term. CRITICAL: if context['obj_correlation'] is empty (check "
        "this explicitly first), this term must contribute EXACTLY 0 to "
        "every candidate's score — do NOT raise an error, do NOT iterate "
        "over hypothetical 'name_a,name_b' keys that aren't actually "
        "present, and do NOT treat an empty dict as evidence of zero "
        "correlation (it means 'no signal available', not 'no "
        "correlation exists'). Your code must run correctly on both "
        "empty-dict and populated-dict inputs.",

    # --- Improvement-direction momentum (v4.1 addition, replaces
    #     phase_decaying_ucb) ---
    # Requires: context["Y_obs"] with at least 4 rows (falls back to
    # acq_value_norm alone otherwise — very early in a campaign there's
    # nothing to split into an older/newer trend yet).
    "improvement_momentum":
        "Reward candidates whose predicted objective vector points in the "
        "SAME DIRECTION the campaign has recently been improving in — a "
        "temporal signal, distinct from every other hint here (none of "
        "them use the ORDER of context['Y_obs'], only its final "
        "contents). First check len(context['Y_obs']) >= 4; if not, skip "
        "this term entirely and score by context['pool'][i]"
        "['acq_value_norm'] alone (too few observations to split "
        "meaningfully). Otherwise, split context['Y_obs'] (all-maximise "
        "convention, append-ordered) into an OLDER half and a NEWER half "
        "by row index (e.g. first half of rows vs second half), compute "
        "each half's per-objective MEAN, and set momentum_direction = "
        "newer_mean - older_mean (a vector in objective space — the "
        "direction the campaign's typical outcome has been drifting). "
        "For each candidate, take (its own gp_posterior mean vector - "
        "newer_mean) and score it by the DOT PRODUCT with "
        "momentum_direction, normalised by momentum_direction's own norm "
        "(if that norm is ~0 — e.g. the campaign hasn't moved recently — "
        "this term contributes 0, do not divide by zero). This rewards "
        "continuing a direction that has recently been paying off. Blend "
        "as a SMALLER secondary term added to "
        "context['pool'][i]['acq_value_norm'], same pattern as the other "
        "additive hints above, not as the sole score.",
}
