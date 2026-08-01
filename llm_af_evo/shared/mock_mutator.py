"""
mock_mutator.py — deterministic, deliberately weak stand-in for the LLM's
crossover/mutation step, so the evolution loop is fully testable in this
sandbox (no Ollama/network access here) before you run it with a real LLM
on your machine.

Programs are represented as {term_name: weight} dicts over a small fixed
term library (af-grammar), not as raw text — mutation/crossover operate on
this structured representation and render_program() turns it into actual
score_pool source. This guarantees every mock-generated program is valid
Python (random text mutation of source code would not have that guarantee),
while still being a genuinely random structural search: term selection and
weights are uniform-random, with no informed guidance toward what wins.
This weakness is intentional — per the earlier design decision, a mock that
beats EGBO-novelty on its own would undermine the point of testing whether
an LLM's mechanism-aware mutation contributes anything.

The real-Ollama path (wired in evolve_af.py, identical in spirit to how
strategy_mo_llm/llm_warmstart.py already call ollama.chat for this project)
produces raw code text directly instead of going through this term-weight
representation, since a real LLM can write arbitrary score_pool bodies
(within the sandbox's import whitelist) — the two code paths converge only
at "produces a Python source string implementing score_pool".
"""

import numpy as np

# term_name -> per-candidate scalar expression string, evaluated inside
# render_program's generated function body, inside a `for cand in
# context["pool"]:` loop where these names are already bound (once, before
# the loop, for anything that doesn't vary per candidate):
#   cand (this candidate's {"x":..., "gp_posterior":...} entry)
#   gp = cand["gp_posterior"]  (rebound each iteration)
#   X_obs, front_range (= context["pareto_front_range"]), progress, stagnant,
#   names (= context["objective_names"])
# Named dict access via a generic `names` loop (gp[name]["mean"], not
# hardcoded gp["Tm"]["mean"] or a positional array column) is deliberate —
# see af_interface.py's contract docstring for why named access matters,
# and evolve_af_v2.py's --oracle coatings support for why it must be
# GENERIC named access, not hardcoded excipient names: mock mode is the
# ONLY candidate-generation path that doesn't go through an LLM (which can
# already read context["objective_names"] itself from the system prompt),
# so if TERM_LIBRARY hardcodes Tm/kD/viscosity, every mu_sum/mu_min/
# sigma_*-based mock candidate KeyErrors on every batch against a
# non-excipient oracle (e.g. coatings' conductivity/conductance_std) —
# discovered directly: a --oracle coatings mock run's evolved population
# never improved past its single hand-written seed, because 6 of 8 terms
# were silently crippled the whole run (see evolve_af_v2.py's module
# history for the diagnosis). Written as sum()/min()/max() generator
# expressions over `names` instead of a literal +/hardcoded-key chain —
# mathematically IDENTICAL to the old hardcoded form for excipient (same
# three terms summed, order doesn't affect +), so this is not a behavior
# change for any existing excipient-oracle run, only a fix for oracles
# with a different objective set.
# The last two terms are phase-aware (scalar-times-scalar): they multiply
# an explore/exploit-relevant term by a campaign-state scalar, giving the
# mock mutator at least some structural access to the same phase-dependent
# strategies a real LLM could express — it's still a uniform-random,
# uninformed search over which terms/weights to pick, so it stays a weak
# baseline; the point is it isn't STRUCTURALLY blind to phase the way a
# purely mu/sigma/novelty term library would be.
TERM_LIBRARY = {
    "mu_sum": "sum(gp[name]['mean'] for name in names)",
    "mu_min": "min(gp[name]['mean'] for name in names)",
    "sigma_sum_norm": "sum(gp[name]['std']/front_range[name] for name in names)",
    "sigma_max_norm": "max(gp[name]['std']/front_range[name] for name in names)",
    "novelty_min": "float(np.linalg.norm(X_obs - cand['x'], axis=1).min())",
    "novelty_mean": "float(np.linalg.norm(X_obs - cand['x'], axis=1).mean())",
    "sigma_sum_norm_early": (
        "sum(gp[name]['std']/front_range[name] for name in names) "
        "* max(0.0, 1.0 - progress)"),
    "novelty_stagnation": (
        "float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5)"),
}
TERM_NAMES = list(TERM_LIBRARY.keys())

_PROGRAM_HEADER = '''def score_pool(context):
    """{docstring}"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
'''


def render_program(term_weights: dict) -> str:
    """{term_name: weight} -> score_pool source string."""
    if not term_weights:
        # Degenerate empty program is a valid but useless AF (always ties);
        # kept renderable rather than raising, so evolution can discover
        # that dropping to zero terms is bad on its own via fitness, not
        # via a crash.
        term_weights = {"mu_sum": 0.0}
    # Mechanical explainability line, not LLM-authored: mock mode has no
    # model to ask "what does this do", but the term_weights dict already
    # says exactly that, so render it directly rather than leaving
    # score_pool without a docstring (which the sandbox now rejects).
    docstring = "weighted sum of: " + ", ".join(
        f"{term}({weight:.2f})" for term, weight in term_weights.items())
    lines = [_PROGRAM_HEADER.format(docstring=docstring)]
    for term, weight in term_weights.items():
        expr = TERM_LIBRARY[term]
        lines.append(f"        s = s + ({weight:.4f}) * ({expr})\n")
    lines.append("        scores.append(s)\n")
    lines.append("    return scores\n")
    return "".join(lines)


def random_program(rng: np.random.Generator, n_terms: int = None) -> dict:
    """Uniform-random term selection and weights — the deliberately weak baseline."""
    if n_terms is None:
        n_terms = int(rng.integers(1, 4))  # 1-3 terms
    chosen = list(rng.choice(TERM_NAMES, size=min(n_terms, len(TERM_NAMES)), replace=False))
    return {t: float(rng.uniform(0.1, 5.0)) for t in chosen}


def mutate(term_weights: dict, rng: np.random.Generator) -> dict:
    """One of: add a random term, drop a random term, or perturb one weight."""
    new = dict(term_weights)
    op = rng.choice(["add", "drop", "perturb"])
    if op == "add" or not new:
        available = [t for t in TERM_NAMES if t not in new]
        if available:
            t = rng.choice(available)
            new[t] = float(rng.uniform(0.1, 5.0))
    elif op == "drop" and len(new) > 1:
        t = rng.choice(list(new.keys()))
        del new[t]
    else:  # perturb
        if new:
            t = rng.choice(list(new.keys()))
            new[t] = float(np.clip(new[t] * rng.uniform(0.5, 2.0), 0.01, 1e6))
    return new


def crossover(a: dict, b: dict, rng: np.random.Generator) -> dict:
    """
    Union of both parents' terms; shared terms average their weights,
    terms unique to one parent are kept as-is (weak, uninformed
    recombination — no fitness-guided term selection).
    """
    child = {}
    for t in set(a) | set(b):
        if t in a and t in b:
            child[t] = float((a[t] + b[t]) / 2.0)
        else:
            child[t] = a.get(t, b.get(t))
    return child
