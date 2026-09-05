"""
evaluate_multi_domain_v6.py — wires evolve_af_v6.py's evaluate_af_2b
(domain-generic via oracle_family, same primitive v5 used) into
multi_domain_fitness_v6.py's z-scored gating/combination logic. Direct
analog of v5's evaluate_multi_domain.py — same dependency-injection
rationale (evaluate_fn passed by the caller, not imported here, avoiding
a circular import with evolve_af_v6.py and keeping v6 self-contained per
this project's versioning convention).
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
    _LLM_AF_EVO / "v6" / "src",
):
    sys.path.insert(0, str(_p))

from multi_domain_fitness_v6 import evaluate_multi_domain_v6, combine_domain_z_scores  # noqa: F401


def build_domain_evaluators_v6(code: str, domain_configs: dict, evaluate_fn, log_dir=None) -> dict:
    """Returns an ORDERED {domain_name: callable() -> (ci_lower_16, se)}
    dict — same shape/contract as v5's build_domain_evaluators. gamma=0 is
    hardcoded in the call here (not evaluate_fn's own adaptive default),
    same reasoning as v5: the complexity penalty is applied ONCE, on the
    final combined fitness, not per-domain.

    domain_configs: {domain_name: {"training_logs": [...], "baseline_hvs":
    [...], "oracle_family": str, "n_fitness_seeds": int (default 1)}} —
    identical shape to v5's.

    evaluate_fn: the caller's evaluate_af_2b (evolve_af_v6.py's own),
    called with af_function_name/combine_with_baseline_acq already bound
    by the caller — this module doesn't know or care about the delta-seed
    contract detail, it only consumes evaluate_fn's (ci_lower_16, se)
    return shape, same as v5's integration layer.
    """
    evaluators = {}
    for domain_name, cfg in domain_configs.items():
        domain_log_dir = (pathlib.Path(log_dir) / domain_name) if log_dir is not None else None

        def _make_fn(domain_name=domain_name, cfg=cfg, domain_log_dir=domain_log_dir):
            def _fn():
                result = evaluate_fn(
                    code, cfg["training_logs"], cfg["baseline_hvs"],
                    gamma=0, oracle_family=cfg["oracle_family"],
                    n_fitness_seeds=cfg.get("n_fitness_seeds", 1),
                    log_dir=domain_log_dir,
                )
                return result["ci_lower_16"], result["se_mean_margin"]
            return _fn

        evaluators[domain_name] = _make_fn()
    return evaluators


def evaluate_candidate_multi_domain_v6(code: str, domain_configs: dict, evaluate_fn,
                                        log_dir=None, gate_z_threshold=None) -> tuple:
    """Convenience wrapper: build_domain_evaluators_v6 +
    evaluate_multi_domain_v6 in one call. Returns (fitness, domain_scores,
    domain_zs, gate_failed_at) — see multi_domain_fitness_v6.
    evaluate_multi_domain_v6's own docstring for the 4-tuple shape (new
    domain_zs element vs v5's 3-tuple)."""
    kwargs = {}
    if gate_z_threshold is not None:
        kwargs["gate_z_threshold"] = gate_z_threshold
    evaluators = build_domain_evaluators_v6(code, domain_configs, evaluate_fn, log_dir=log_dir)
    return evaluate_multi_domain_v6(evaluators, **kwargs)
