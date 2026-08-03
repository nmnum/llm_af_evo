"""
controllers/approach_a.py — Approach A: LLM tunes UCB β only.

The LLM receives campaign context and outputs a single scalar β ∈ [0.01, 1000].
Strategy family is fixed to UCB; only the exploration-exploitation trade-off changes.
"""

import pathlib
from typing import Any, Dict

from controllers.base import OllamaController

_SYSTEM_PROMPT = pathlib.Path(__file__).parent.parent / "prompts" / "system_a.txt"


class ApproachAController(OllamaController):
    """
    Approach A: LLM tunes UCB β.

    Decision output: {"beta": <float>}
    """

    _needs_gp_uncertainty = True

    def __init__(self, model: str = "qwen2.5-coder:7b", **kwargs):
        system = _SYSTEM_PROMPT.read_text() if _SYSTEM_PROMPT.exists() else _DEFAULT_SYSTEM
        super().__init__(model=model, system_prompt=system, **kwargs)
        self._current_beta = 1.0

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        progress = context["step"] / context["budget"]
        return (
            f"Campaign status:\n"
            f"  Step: {context['step']} / {context['budget']} ({100*progress:.0f}% complete)\n"
            f"  Best found (normalised): {context['best_normalised']:.3f}\n"
            f"  Improvement rate (last 10 steps): {context['improvement_rate']:.2f}\n"
            f"  GP uncertainty (mean posterior std): {context['gp_uncertainty']:.4f}\n"
            f"  Current UCB β: {self._current_beta:.2f}\n\n"
            f"Respond with JSON only: {{\"beta\": <float between 0.01 and 1000>}}\n"
            f"Higher β = more exploration. Lower β = more exploitation.\n"
            f"The β MUST reflect the current campaign phase — it should decrease as progress increases."
        )

    def _parse_response(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        parsed = self._extract_json(text)
        if parsed and "beta" in parsed:
            try:
                beta = float(parsed["beta"])
                beta = max(0.01, min(1000.0, beta))
                self._current_beta = beta
                return {"strategy": "ucb", "params": {"beta": beta}}
            except (ValueError, TypeError):
                pass
        return {"strategy": "ucb", "params": {"beta": self._current_beta}}


_DEFAULT_SYSTEM = """You are an autonomous optimisation controller for a self-driving laboratory.
Your task is to tune the exploration-exploitation trade-off of a UCB Bayesian optimisation strategy
by choosing the β parameter. Respond with JSON only."""
