"""
controllers/approach_a.py — Approach A: LLM tunes UCB β only.

The LLM receives campaign context and outputs a single scalar β ∈ [0.01, 1000].
Strategy family is fixed to UCB; only the exploration-exploitation trade-off changes.
"""

import pathlib
from typing import Any, Dict, Optional

from controllers.base import OllamaController

_SYSTEM_PROMPT = pathlib.Path(__file__).parent.parent / "prompts" / "system_a.txt"


class ApproachAController(OllamaController):
    """
    Approach A: LLM tunes UCB β.

    Decision output: {"beta": <float>}
    """

    _needs_gp_uncertainty = True

    def __init__(self, model: str = "qwen3-coder:30b", **kwargs):
        system = _SYSTEM_PROMPT.read_text() if _SYSTEM_PROMPT.exists() else _DEFAULT_SYSTEM
        kwargs.setdefault("num_predict", 64)  # answer is ~10 tokens: {"beta": 5.0}
        super().__init__(model=model, system_prompt=system, **kwargs)
        self._current_beta = 1.0
        # Outcome feedback: was the last beta actually working? Without this the
        # LLM only ever sees "what phase am I in", never "did my last choice help".
        self._last_best_normalised: Optional[float] = None

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        progress = context["step"] / context["budget"]
        best_now = context["best_normalised"]

        if self._last_best_normalised is not None:
            delta = best_now - self._last_best_normalised
            feedback = f"  Since the last β choice, best_normalised changed by {delta:+.4f}.\n"
        else:
            feedback = "  This is the first decision — no outcome feedback yet.\n"
        self._last_best_normalised = best_now

        return (
            f"Campaign status:\n"
            f"  Step: {context['step']} / {context['budget']} ({100*progress:.0f}% complete)\n"
            f"  Best found (normalised): {best_now:.3f}\n"
            f"  Improvement rate (last 10 steps): {context['improvement_rate']:.2f}\n"
            f"  GP uncertainty (mean posterior std): {context['gp_uncertainty']:.4f}\n"
            f"  Current UCB β: {self._current_beta:.2f}\n"
            f"{feedback}\n"
            f"Respond with JSON only: {{\"beta\": <float between 0.01 and 1000>}}\n"
            f"Higher β = more exploration. Lower β = more exploitation.\n"
            f"The β MUST reflect the current campaign phase — it should decrease as progress increases.\n"
            f"If the last β choice produced no improvement, don't just repeat it — use the "
            f"feedback above to move further along the exploration/exploitation axis."
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
