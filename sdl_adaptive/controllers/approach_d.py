"""
controllers/approach_d.py — LLM strategy router (Approach D).

The LLM receives campaign-state signals and selects a strategy + optional parameter.
Uses qwen2.5-coder:7b-instruct (NOT the coder model) for better instruction following.

Output format — flat JSON, two fields maximum:
  {"strategy": "egbo"}
  {"strategy": "lhs"}
  {"strategy": "ucb", "beta": 2.0}

This collapses strategy selection and parameter tuning into one call.
Beta is only parsed when strategy == "ucb".

Key design decisions:
  - Uses the instruct model, not the coder model, for discrete routing
  - Falls back to rule_router logic if LLM output doesn't parse
  - Tracks consecutive stagnation to detect if LLM is ignoring signals
  - _needs_gp_uncertainty = True to get lengthscale_norm in context
"""

import json
import logging
import pathlib
import re
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = pathlib.Path(__file__).parent.parent / "prompts" / "system_d.txt"

VALID_STRATEGIES = {"egbo", "novelty_egbo", "ucb", "lhs", "random"}

# Fallback thresholds (mirrors rule_router, used when LLM fails)
_LS_ROUGH   = 0.15
_LS_SMOOTH  = 0.30
_BEST_EXPLOIT = 0.90
_MIN_OBS_PER_DIM = 3.0
_STAGNATION = 0.05


def _rule_fallback(context: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback when LLM output is unparseable."""
    ls       = context.get("lengthscale_norm", 0.5)
    best     = context["best_normalised"]
    improv   = context["improvement_rate"]
    n_obs    = context["n_obs"]
    n_dims   = context.get("n_dims", context["bounds"].shape[0])

    if best > _BEST_EXPLOIT:
        return {"strategy": "ucb", "params": {"beta": 0.1}}
    if n_obs / max(n_dims, 1) < _MIN_OBS_PER_DIM:
        return {"strategy": "lhs", "params": {}}
    if ls < _LS_ROUGH:
        return {"strategy": "lhs", "params": {}}
    if improv < _STAGNATION:
        return {"strategy": "egbo", "params": {}}
    if ls > _LS_SMOOTH:
        return {"strategy": "egbo", "params": {}}
    # UCB with progress-decayed beta
    progress = context["step"] / context["budget"]
    beta = max(0.5, 10.0 * (1.0 - progress))
    return {"strategy": "ucb", "params": {"beta": round(beta, 2)}}


class ApproachDController:
    """
    LLM strategy router. One call per decision point.

    Parameters
    ----------
    model        : Ollama model name. Default: qwen2.5-coder:7b-instruct
                   Use the instruct variant, not the base coder model.
    max_retries  : Number of retries on parse failure (default 2).
    timeout      : Seconds to wait for Ollama response (default 30).
    """

    _needs_gp_uncertainty = True  # triggers lengthscale_norm computation

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b-instruct",
        max_retries: int = 2,
        timeout: int = 30,
    ):
        self.model       = model
        self.max_retries = max_retries
        self.timeout     = timeout

        self._system = (
            _SYSTEM_PROMPT.read_text() if _SYSTEM_PROMPT.exists()
            else _DEFAULT_SYSTEM
        )
        self._current_strategy = "lhs"
        self._current_params: Dict = {}
        self._consecutive_same = 0

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        n_dims   = context.get("n_dims", context["bounds"].shape[0])
        obs_per_dim = context["n_obs"] / max(n_dims, 1)
        progress = context["step"] / context["budget"]
        ls       = context.get("lengthscale_norm", 0.5)

        return (
            f"best_normalised:  {context['best_normalised']:.3f}\n"
            f"improvement_rate: {context['improvement_rate']:.3f}\n"
            f"gp_uncertainty:   {context['gp_uncertainty']:.4f}\n"
            f"lengthscale_norm: {ls:.3f}\n"
            f"obs_per_dim:      {obs_per_dim:.1f}\n"
            f"progress:         {progress:.2f}\n"
            f"current_strategy: {self._current_strategy}"
            + (f" (selected {self._consecutive_same} times in a row — "
               f"consider whether it's still appropriate)" if self._consecutive_same >= 3 else "")
            + "\n\n"
            f"Select the best strategy. Output JSON only."
        )

    def _parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract strategy + optional beta from flat JSON response."""
        # Strip qwen3 thinking tokens (chain-of-thought before actual answer)
        import re as _re
        text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()

        # Strip markdown fences
        clean = text.strip()
        for fence in ["```json", "```"]:
            if clean.startswith(fence):
                clean = clean[len(fence):]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        # Try direct parse first
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            # Try extracting first JSON object
            m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if not m:
                return None
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                return None

        strategy = str(parsed.get("strategy", "")).lower().strip()
        if strategy not in VALID_STRATEGIES:
            return None

        params = {}
        if strategy == "ucb":
            beta = parsed.get("beta", parsed.get("params", {}).get("beta", 2.0))
            try:
                params["beta"] = float(max(0.01, min(400.0, float(beta))))
            except (ValueError, TypeError):
                params["beta"] = 2.0

        return {"strategy": strategy, "params": params}

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call LLM and return routing decision."""
        try:
            import ollama
        except ImportError:
            logger.warning("ollama not installed — using rule fallback")
            return _rule_fallback(context)

        prompt = self._build_prompt(context)

        for attempt in range(self.max_retries):
            try:
                # think=False disables chain-of-thought for qwen3 models
                # num_predict=256 gives enough tokens after thinking is stripped
                options = {"temperature": 0.1, "num_predict": 256}
                if "qwen3" in self.model.lower():
                    options["think"] = False
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._system},
                        {"role": "user",   "content": prompt},
                    ],
                    options=options,
                )
                text = response["message"]["content"]
                decision = self._parse(text)
                if decision is not None:
                    # Track consecutive same-strategy selections
                    if decision["strategy"] == self._current_strategy:
                        self._consecutive_same += 1
                    else:
                        self._consecutive_same = 1
                    self._current_strategy = decision["strategy"]
                    self._current_params   = decision["params"]
                    return decision

                logger.warning(
                    f"ApproachD attempt {attempt+1}: unparseable response: "
                    f"{text[:100]!r}"
                )

            except Exception as e:
                logger.warning(f"ApproachD attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)

        # All retries failed — use rule-based fallback
        logger.warning("ApproachD: all retries failed, using rule fallback")
        decision = _rule_fallback(context)
        self._current_strategy = decision["strategy"]
        self._current_params   = decision["params"]
        return decision


_DEFAULT_SYSTEM = """\
Select the best optimisation strategy given the campaign state.
Output flat JSON only: {"strategy": "<name>"} or {"strategy": "ucb", "beta": <float>}
Strategies: egbo, ucb, lhs, random
"""
