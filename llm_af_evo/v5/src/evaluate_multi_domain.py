"""
evaluate_multi_domain.py — wires real per-domain evaluation (v4's
evaluate_af_2b, already domain-generic via oracle_family) into
multi_domain_fitness.py's gating/combination logic. Kept separate from
both of those, and from evolve_af_v5.py's evolution loop, so this
integration layer can be verified against real training data one domain
at a time before any full-run wiring.

The per-domain single-evaluation function (e.g. evolve_af_v5.py's own
evaluate_af_2b — bootstrap ci_lower_16, sandbox replay via
full_replay.run_2b_campaign, already domain-generic via oracle_family) is
passed in by the CALLER (build_domain_evaluators' evaluate_fn param), not
imported here — this module never hardcodes which version's evaluate_af_2b
it uses. Two reasons: (1) it avoids a circular import (evolve_af_v5.py
needs to call INTO this module for the gating/combination logic, so this
module can't also import evolve_af_v5.py's evaluate_af_2b at module load
time), and (2) it keeps v5 self-contained per this project's own
versioning convention (see af_interface_v3.py's docstring on why
STRATEGY_HINTS is copied, not shared across releases) — evolve_af_v5.py
passes ITS OWN evaluate_af_2b, not v4's, so this module has no dependency
on any specific version's evolution file at all.
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
    _LLM_AF_EVO / "v5" / "src",
):
    sys.path.insert(0, str(_p))

from multi_domain_fitness import evaluate_multi_domain, combine_domain_scores  # noqa: F401 (re-exported)


def build_domain_evaluators(code: str, domain_configs: dict, evaluate_fn, log_dir=None) -> dict:
    """Returns an ORDERED {domain_name: callable() -> (ci_lower_16, se)}
    dict, ready to pass straight to multi_domain_fitness.evaluate_multi_domain
    — order of domain_configs IS the gate sequence (caller's
    responsibility; e.g. an OrderedDict/plain dict literal with tunable
    first, mAb last).

    domain_configs: {domain_name: {"training_logs": [...], "baseline_hvs":
    [...], "oracle_family": str, "n_fitness_seeds": int (default 1)}} —
    same shapes evaluate_fn already expects per-domain (see
    evolve_af_v5.py's own run_evolution setup for how training_logs/
    baseline_hvs are loaded and paired per oracle_family).

    evaluate_fn: the caller's single-domain evaluator, called as
    evaluate_fn(code, training_logs, baseline_hvs, gamma=0,
    oracle_family=..., n_fitness_seeds=..., log_dir=...) -> dict with
    "ci_lower_16" and "se_mean_margin" keys (evolve_af_v5.py's
    evaluate_af_2b's exact signature/return shape). gamma=0 is hardcoded
    in the call here, not left to evaluate_fn's own adaptive default —
    per the design conversation's decision: the complexity penalty is
    applied ONCE, on the final combined multi-domain fitness
    (evolve_af_v5.py's job), not per-domain, which would triple/
    quadruple-penalize the same LOC count once per domain evaluated.
    Each callable returns (ci_lower_16, se_mean_margin) — the raw
    bootstrap-CI-lower-bound margin, noise-aware, plus its own standard
    error, which multi_domain_fitness.combine_domain_scores_weighted
    uses to inverse-variance-weight the final combination (see its own
    docstring for why: tunable/coatings/mAb's very different intrinsic
    noise levels mean their ci_lower_16 values are NOT equally
    trustworthy, and treating them as if they were let whichever domain
    happened to be noisiest dominate the cross-domain consistency
    penalty). Not evaluate_fn's own `fitness` field (which would already
    have a zero-gamma no-op penalty subtracted — using ci_lower_16
    directly is equivalent here since gamma=0, but named explicitly for
    clarity if that ever changes).

    log_dir: if given, each domain's evaluate_fn call logs candidate code
    into a PER-DOMAIN subdirectory (log_dir/domain_name), so replay logs
    from different domains never collide or get attributed to the wrong
    domain.
    """
    evaluators = {}
    for domain_name, cfg in domain_configs.items():
        domain_log_dir = (pathlib.Path(log_dir) / domain_name) if log_dir is not None else None

        def _make_fn(domain_name=domain_name, cfg=cfg, domain_log_dir=domain_log_dir):
            # Explicit default-arg binding — closes over THIS iteration's
            # domain_name/cfg/domain_log_dir, not the loop variable's
            # final value (the classic late-binding closure bug).
            def _fn():
                result = evaluate_fn(
                    code, cfg["training_logs"], cfg["baseline_hvs"],
                    gamma=0, oracle_family=cfg["oracle_family"],
                    n_fitness_seeds=cfg.get("n_fitness_seeds", 1),
                    log_dir=domain_log_dir,
                )
                # (ci_lower_16, se_mean_margin) pair — see
                # multi_domain_fitness.evaluate_multi_domain's docstring:
                # se is used only for the final inverse-variance-weighted
                # combination among domains that all pass their gates,
                # not for the gate comparison itself.
                return result["ci_lower_16"], result["se_mean_margin"]
            return _fn

        evaluators[domain_name] = _make_fn()
    return evaluators


def evaluate_candidate_multi_domain(code: str, domain_configs: dict, evaluate_fn,
                                     log_dir=None, lam=None, gate_threshold=None):
    """Convenience wrapper: build_domain_evaluators + evaluate_multi_domain
    in one call. kwargs not explicitly passed use multi_domain_fitness's
    own defaults (LAMBDA_STD_PENALTY, GATE_THRESHOLD)."""
    kwargs = {}
    if lam is not None:
        kwargs["lam"] = lam
    if gate_threshold is not None:
        kwargs["gate_threshold"] = gate_threshold
    evaluators = build_domain_evaluators(code, domain_configs, evaluate_fn, log_dir=log_dir)
    return evaluate_multi_domain(evaluators, **kwargs)
