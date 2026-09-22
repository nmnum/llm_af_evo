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

---
v6: DELTA-SEED CONTRACT — everything below this point is a v6-specific
rewrite, not a copy of v5's. The rest of this module docstring (v2/v4
history above) describes the OLD full-replacement score_pool contract
this file inherited by copying af_interface_v5.py; v6 changes the
contract itself, so the seed/hint catalog below no longer matches it.

v6's diagnostic (see v6/README.md): across v5's two full 150-generation
runs, only 1 of 8 "fitness improved" events was distinguishable from
measurement noise — training-time selection was mostly navigating noise,
not genuine mechanism quality, at v5's campaign counts and substrate.
Separately: every real champion v2-v5 ever found, on inspection, turned
out to be "acq_value_norm (real qLogNEHVI) plus a small correction" —
the LLM kept re-deriving this shape inside an unconstrained rewrite
every time. v6 formalises that shape as the search space directly.

Contract change: candidate code defines `modifier(context)`, NOT
`score_pool(context)` — see sandbox.py's af_function_name/
combine_with_baseline_acq params (run_af_in_sandbox is called with
af_function_name="modifier", combine_with_baseline_acq=True for every
v6 evaluation). `modifier(context)` returns ONE VALUE PER CANDIDATE (same
list-of-floats shape as score_pool always did) — but that value is a
CORRECTION TERM ONLY. The sandbox itself (not the candidate code, not
this module) adds context["pool"][i]["acq_value_norm"] to each returned
value before validating/returning the final score. A candidate that
returns all zeros is therefore STRUCTURALLY equivalent to EGBO's raw
acq_value_norm baseline — verified directly (see sandbox.py's own test
coverage) — not merely conventionally close to it.

This means every hint/seed below must NOT re-add acq_value_norm itself
(the harness already does that) and must NOT try to reconstruct a full
score — writing `return acq_value_norm + something` inside modifier()
would double-count the baseline. The context dict is otherwise
UNCHANGED from v5 (context["pool"][i]["acq_value_norm"] is still
present and readable, e.g. for a modifier that wants to scale itself by
current acquisition strength) — only what the return value MEANS
changed, not what's available to compute it from.

alpha (the modifier's own scale) is not a separate extracted parameter
— it's just an ordinary literal float the LLM writes directly into its
own expression (e.g. `return list(0.3 * uncertainty_term)`), exactly
like every other tunable constant AF code has always set for itself.
Additive combination was chosen over multiplicative/gated alternatives
for interpretability and to avoid hand-rolled branching — a documented
LLM failure mode across v2-v5 (see v6/README.md's decision 2).
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

from af_interface import (  # noqa: F401 — re-exported for evolve_af_v6.py
    ALLOWED_GLOBALS, OBJECTIVE_NAMES,
    select_batch, count_loc, extract_af_docstring, has_return_statement,
)

# v6's delta-seed contract uses a DIFFERENT top-level function name than
# every prior version's AF_FUNCTION_NAME ("score_pool") — this is a
# structural change, not a naming convention, so it gets its own
# constant rather than shadowing af_interface.py's. Every v6 call site
# (sandbox.py's run_af_in_sandbox, count_loc/has_return_statement/
# extract_af_docstring) must pass af_function_name=AF_FUNCTION_NAME_V6
# explicitly — there is no global default that would make this silently
# correct, by design (a v6 evolution loop that forgets to pass it would
# get a loud "modifier is missing a required docstring"-style failure
# immediately, not a silent fallback to checking for "score_pool").
AF_FUNCTION_NAME_V6 = "modifier"
COMBINE_WITH_BASELINE_ACQ_V6 = True

# The one full hand-written seed for v6 — plays the same anchoring role
# trust_only/egbo_novelty_like played in v2-v5 (demonstrates the exact
# contract syntax an LLM must match), but necessarily a NEW program:
# every prior seed wrote a full score_pool, which is no longer a valid
# v6 program (writing `return acq_value_norm + ...` inside modifier()
# would double-count the baseline the sandbox already adds).
#
# uncertainty_modifier: a classic UCB-style exploration term — reward
# candidates the GP posterior is still uncertain about, scaled by how
# far through the campaign budget we are (heavier early, lighter late,
# same progress-decay idea v5's gen81_jitter1 lineage used, chosen here
# specifically because it's simple, well-understood, and safe as an
# anchor — not because it's expected to win; that's evolution's job).
SEED_UNCERTAINTY_MODIFIER = '''def modifier(context):
    """Progress-decaying uncertainty bonus: rewards candidates the GP
    posterior is still uncertain about, more heavily early in the
    campaign. Returns ONLY this correction term — acq_value_norm is
    added automatically by the sandbox, not by this function."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    weight = 0.3 * (1.0 - progress)
    terms = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names)
        terms.append(weight * sigma_sum)
    return terms
'''


SEED_PROGRAMS = {
    "uncertainty_modifier": SEED_UNCERTAINTY_MODIFIER.strip("\n"),
}
# No SEED_TERM_WEIGHTS/mock-mode term-weight representation for v6 — the
# whole point of the delta-seed contract is that alpha/mechanism choice
# needs real reasoning about what correction term to try, which mock
# mode's fixed TERM_TEMPLATES vocabulary (designed for full score_pool
# rewrites) has no analog for. v6 is real-LLM-only; evolve_af_v6.py does
# not implement a mock-mode fallback path at all (see v6/README.md).

# Strategy hints (v6): 6 distinct MODIFIER mechanisms — deliberately a
# SMALLER catalog than v5's 10, because every hint here has to satisfy a
# narrower brief ("write a bounded correction term added to
# acq_value_norm automatically") rather than "write a full scoring
# function", so several v5 hints collapse into variants of the same
# thing once the "must include/blend with acq_value_norm yourself"
# instruction is removed (the sandbox does that now). Hints reused from
# v5 are trimmed to drop that now-automatic instruction and to
# explicitly warn against re-adding acq_value_norm inside modifier()
# itself (double-counting the baseline) — everything else (correct
# dominance-test direction, no hand-rolled HV, O(n) not O(n^2) cost
# under the sandbox's 10s timeout) carries over unchanged from v5's own
# hard-won hint-rewrite history.
#
# Hints requiring context["Y_obs"]: outcome_novelty, improvement_momentum
# Hints requiring context["pareto_front"]: front_coverage_gap
# Hints requiring neither: dpp_diversity_bonus, local_penalization_bonus,
#                           stagnation_boost
STRATEGY_HINTS = {
    # --- Outcome-space novelty bonus (trimmed from v5's outcome_novelty) ---
    "outcome_novelty":
        "Return a bonus rewarding candidates whose predicted outcome is "
        "far from anything already observed: for each candidate, compare "
        "its gp_posterior mean vector to every row of context['Y_obs'] "
        "(the full observation history, not context['pareto_front'] "
        "alone) and return a value proportional to its nearest-neighbor "
        "distance in objective space, min-max normalised across the pool "
        "to keep the term's own scale bounded and comparable across "
        "generations (e.g. into roughly [0, 0.5] via an explicit scale "
        "factor you choose). Do NOT add context['pool'][i]['acq_value_norm'] "
        "yourself anywhere in this function — the sandbox adds it "
        "automatically after modifier() returns; this function returns "
        "ONLY the novelty correction.",

    # --- Diversity bonus (trimmed from v5's dpp_diversity, dropped the
    #     multiplicative-quality-times-diversity framing since quality
    #     is now handled entirely by the automatic acq_value_norm add) ---
    "dpp_diversity_bonus":
        "Return a bonus for candidates that are mutually DISSIMILAR from "
        "other high-acq_value_norm candidates in the pool, discouraging "
        "the batch from clustering on near-duplicates: build a "
        "similarity kernel between candidates (e.g. exp(-distance) in x "
        "or in gp_posterior mean space, EXCLUDING each candidate's "
        "similarity to itself) and return a value that is LOW when a "
        "candidate closely resembles other candidates that also have "
        "high acq_value_norm (context['pool'][i]['acq_value_norm'] is "
        "readable here — you may READ it to decide who counts as "
        "'other high-scoring candidates', you must simply not ADD it "
        "into your own return value) and HIGH when it doesn't. Scale the "
        "final bonus to a bounded range (e.g. [0, 0.3]) via an explicit "
        "factor you choose.",

    # --- Local-penalization-style bonus (trimmed from v5's
    #     local_penalization; same "multiplier, never (1-multiplier)"
    #     warning carried over verbatim — a repeated real failure mode) ---
    "local_penalization_bonus":
        "Return a bonus that greedily discourages picking near-duplicate "
        "candidates within the SAME batch: rank the pool by "
        "context['pool'][i]['acq_value_norm'] first (read-only — do not "
        "modify or return this value directly), then walk down that "
        "ranking and, for each candidate, compute a `multiplier` in "
        "[0, 1] that is LOW (near 0) when the candidate is CLOSE (in x) "
        "to a higher-ranked candidate already walked, and HIGH (near 1) "
        "when it is far from all of them — e.g. "
        "`multiplier = 1.0 - exp(-dist)`, used exactly as-is. Return "
        "`(multiplier - 1.0) * small_scale` (a NEGATIVE-or-zero "
        "correction — penalizing clustered candidates relative to the "
        "baseline they'd otherwise get, never a positive bonus for being "
        "clustered) with small_scale an explicit bounded constant you "
        "choose (e.g. 0.3). Never write `1 - multiplier` anywhere else in "
        "this function; `multiplier` itself already has the correct "
        "sense (low=penalize, high=leave alone).",

    # --- Front coverage gap (trimmed from v5's front_coverage_gap) ---
    "front_coverage_gap":
        "Return a bonus for candidates predicted to land in an "
        "UNDER-COVERED region of the current Pareto front: for each "
        "candidate, take its gp_posterior mean vector and compute its "
        "distance to the k=3 nearest points already in "
        "context['pareto_front'] (fall back to context['Y_obs'] if the "
        "front has fewer than 3 points — very early in a campaign). "
        "Return the MEAN of those k nearest-distances, scaled by an "
        "explicit bounded factor you choose (e.g. 0.2-0.4): a candidate "
        "near a sparse stretch of the front should get a larger bonus "
        "than one near a dense cluster, even if both are equally close "
        "to the front overall. Cost is O(n_candidates * n_front_points) "
        "— cheap at typical front sizes, well under the sandbox's "
        "10-second timeout.",

    # --- Improvement-direction momentum (trimmed from v5's
    #     improvement_momentum) ---
    "improvement_momentum":
        "Return a bonus rewarding candidates whose predicted outcome "
        "points in the SAME DIRECTION the campaign has recently been "
        "improving in. First check len(context['Y_obs']) >= 4; if not, "
        "return all zeros (too few observations to split meaningfully — "
        "acq_value_norm alone, added automatically, is the whole score "
        "in that case). Otherwise split context['Y_obs'] (append-ordered) "
        "into an OLDER half and a NEWER half by row index, compute each "
        "half's per-objective MEAN, and set momentum_direction = "
        "newer_mean - older_mean. For each candidate, take (its "
        "gp_posterior mean vector - newer_mean) and return its DOT "
        "PRODUCT with momentum_direction, normalised by "
        "momentum_direction's own norm (return 0 for every candidate if "
        "that norm is ~0 — do not divide by zero), scaled by an explicit "
        "bounded factor you choose.",

    # --- Stagnation-adaptive exploration boost (new for v6 — not a v5
    #     hint; the noisy-synthetic substrate makes campaign-level
    #     stagnation a more informative signal than in v5's noisier real
    #     domains, where stagnant_batches was harder to trust) ---
    "stagnation_boost":
        "Return a bonus that grows when the campaign has been stagnant "
        "(context['campaign']['stagnant_batches'] is high) and shrinks "
        "when it's actively improving, rewarding candidates the GP is "
        "still uncertain about MORE heavily during a stagnant stretch "
        "(when pure exploitation of acq_value_norm alone has stopped "
        "finding anything new) and LESS heavily otherwise. Compute a "
        "per-candidate uncertainty term (e.g. sum of gp_posterior std "
        "across objectives, normalised by context['pareto_front_range']), "
        "then scale it by a factor that increases with "
        "context['campaign']['stagnant_batches'] (e.g. "
        "`min(stagnant_batches / 5.0, 1.0)`, clipped so it doesn't grow "
        "unbounded over a very long stagnant stretch) times an explicit "
        "bounded base factor you choose (e.g. 0.3).",
}
