"""
controllers/approach_b.py — Approach B: LLM outputs a continuous exploration score,
Python maps it to a strategy.

The core insight from empirical testing: qwen2.5-coder:7b cannot reliably select
from a discrete menu of strategies based on campaign context. It CAN reliably output
calibrated numerical scores (as approach_a demonstrates with beta).

Design: ask the LLM for a single number — an "exploration score" in [0, 1] —
that represents how much exploration the campaign needs right now.
Python maps this score to a strategy:
  score > 0.75  → lhs        (high exploration need)
  score > 0.45  → ucb        (moderate, with beta derived from score)
  score > 0.20  → ei         (exploitation focus)
  score <= 0.20 → pi         (strong exploitation / near-optimal)

The LLM's genuine contribution: reading improvement_rate, gp_uncertainty,
best_normalised, and progress to produce a score that reflects the actual
campaign state — not just the phase.

This makes approach_b meaningfully different from approach_a:
  - approach_a: strategy fixed (UCB), LLM tunes one parameter continuously
  - approach_b: LLM outputs exploration need, Python selects strategy + params
  - approach_c: LLM rewrites the entire optimiser function
"""

import logging
import pathlib
from typing import Any, Dict, Optional

from controllers.base import OllamaController

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = pathlib.Path(__file__).parent.parent / "prompts" / "system_b.txt"

VALID_STRATEGIES = {"ucb", "ei", "pi", "thompson", "random", "lhs"}

# Exploration score → strategy mapping
# Score is in [0, 1]: 1 = maximum exploration, 0 = maximum exploitation
_SCORE_THRESHOLDS = [
    (0.75, "lhs",  {}),
    (0.45, "ucb",  None),   # beta derived from score
    (0.20, "ei",   {}),
    (0.00, "pi",   {}),
]


def _score_to_strategy(score: float) -> Dict[str, Any]:
    """Map a continuous exploration score to a strategy + params."""
    score = float(max(0.0, min(1.0, score)))
    for threshold, strategy, params in _SCORE_THRESHOLDS:
        if score >= threshold:
            if strategy == "ucb":
                # Map score linearly within [0.45, 0.75] → beta in [0.5, 50]
                t = (score - 0.45) / 0.30          # 0 at threshold, 1 at top
                beta = 0.5 * (50.0 / 0.5) ** t     # log-linear: 0.5 → 50
                return {"strategy": "ucb", "params": {"beta": round(beta, 2)}}
            return {"strategy": strategy, "params": dict(params) if params else {}}
    return {"strategy": "pi", "params": {}}


class ApproachBController(OllamaController):
    """
    Approach B: LLM outputs exploration score [0,1], Python selects strategy.
    """

    _needs_gp_uncertainty = True

    def __init__(self, model: str = "qwen3-coder:30b", **kwargs):
        system = _SYSTEM_PROMPT.read_text() if _SYSTEM_PROMPT.exists() else _DEFAULT_SYSTEM
        kwargs.setdefault("num_predict", 64)  # answer is ~10 tokens: {"score": 0.42}
        super().__init__(model=model, system_prompt=system, **kwargs)
        self._current_score = 0.8   # start with exploration
        self._current_strategy = "lhs"
        self._current_params: Dict = {}
        # Outcome feedback: what did the currently-active strategy actually achieve
        # since it was chosen? Without this the LLM never learns whether its picks work.
        self._strategy_set_at_best: Optional[float] = None
        self._strategy_set_at_step: Optional[int] = None

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        progress = context["step"] / context["budget"]
        best_now = context["best_normalised"]

        if self._strategy_set_at_best is not None:
            delta = best_now - self._strategy_set_at_best
            n_steps = context["step"] - self._strategy_set_at_step
            feedback = (
                f"  Last strategy '{self._current_strategy}' ran for {n_steps} step(s); "
                f"best_normalised changed by {delta:+.4f} during that time.\n"
            )
        else:
            feedback = "  This is the first decision — no strategy feedback yet.\n"

        # Show the LLM how scores map to strategies as a reference
        return (
            f"Campaign state:\n"
            f"  Progress: {100*progress:.0f}% (step {context['step']} / {context['budget']})\n"
            f"  Best found (normalised 0–1): {best_now:.3f}\n"
            f"  Improvement rate (last 10 steps): {context['improvement_rate']:.2f}\n"
            f"  GP uncertainty (mean posterior std): {context['gp_uncertainty']:.4f}\n"
            f"  Current score: {self._current_score:.2f} → {self._current_strategy}\n"
            f"{feedback}\n"
            f"Score → strategy mapping:\n"
            f"  0.75 – 1.00 : lhs        (space-filling, maximum exploration)\n"
            f"  0.45 – 0.75 : ucb        (GP-based, beta scales with score)\n"
            f"  0.20 – 0.45 : ei         (expected improvement, exploitation focus)\n"
            f"  0.00 – 0.20 : pi         (probability of improvement, near-optimal)\n\n"
            f"Guidelines:\n"
            f"  - Early campaign, high uncertainty → score near 1.0\n"
            f"  - Mid campaign, moderate progress → score near 0.5–0.6\n"
            f"  - Late campaign, good best found → score near 0.2–0.3\n"
            f"  - Stagnating (improvement_rate < 0.1) → increase score by 0.2\n"
            f"  - best_normalised > 0.95 → score near 0.1\n"
            f"  - If the feedback above shows the current strategy produced little/no "
            f"improvement over several steps, don't just repeat the same score — move it.\n\n"
            f"Return a single exploration score in [0.0, 1.0] that reflects the "
            f"current campaign state. The score should decrease as the campaign matures.\n\n"
            f"Respond with JSON only: {{\"score\": <float between 0.0 and 1.0>}}"
        )

    def _parse_response(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        parsed = self._extract_json(text)
        if parsed and "score" in parsed:
            try:
                score = float(parsed["score"])
                score = max(0.0, min(1.0, score))
                self._current_score = score
                decision = _score_to_strategy(score)
                self._set_strategy(decision, context)
                return decision
            except (ValueError, TypeError):
                pass

        # Fallback: decay score based on progress. This is a reasonable decision on
        # its own, but it means a parse failure is invisible in the results — log it
        # so a run's LLM-success-rate can be checked after the fact.
        logger.warning(
            f"ApproachB: LLM response unparseable ({text[:100]!r}) — "
            f"using progress-decay fallback score."
        )
        progress = context["step"] / context["budget"]
        fallback_score = max(0.1, 1.0 - progress)
        self._current_score = fallback_score
        decision = _score_to_strategy(fallback_score)
        self._set_strategy(decision, context)
        return decision

    def _set_strategy(self, decision: Dict[str, Any], context: Dict[str, Any]) -> None:
        self._current_strategy = decision["strategy"]
        self._current_params = decision["params"]
        self._strategy_set_at_best = context["best_normalised"]
        self._strategy_set_at_step = context["step"]


_DEFAULT_SYSTEM = """You are an autonomous optimisation controller for a self-driving laboratory.
Output a single exploration score in [0, 1] reflecting how much exploration the campaign needs.
Respond with JSON only: {"score": <float>}"""
