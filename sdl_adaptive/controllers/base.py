"""
controllers/base.py — Base class for Ollama-backed LLM controllers.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OllamaController:
    """
    Base class for controllers that call a local Ollama model.

    Subclasses must implement:
      _build_prompt(context) -> str
      _parse_response(text, context) -> dict  (the decision dict)

    Parameters
    ----------
    model        : Ollama model name (e.g. 'qwen2.5-coder:7b')
    system_prompt: system prompt string
    max_retries  : number of retries on parse failure
    timeout      : seconds to wait for Ollama response
    """

    _needs_gp_uncertainty = True  # LLM controllers need GP uncertainty

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        system_prompt: str = "",
        max_retries: int = 3,
        timeout: int = 60,
        temperature: float = 0.7,
        repeat_penalty: float = 1.3,
        num_predict: int = 2048,
        stop: Optional[list] = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.timeout = timeout
        self.stop = stop
        # temperature 0.1 makes code-generation converge to whatever the
        # prompt already shows (i.e. the current code, verbatim) — there's
        # no genuine variation to select on. 0.7 + a repeat_penalty gives
        # the model room to actually rewrite instead of echo.
        # (matches the settings used for the evolved-AF runs, see
        # llm_af_evo/v2/src/evolve_af_v2.py)
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.num_predict = num_predict

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call Ollama and return a decision dict."""
        import ollama

        prompt = self._build_prompt(context)

        for attempt in range(self.max_retries):
            try:
                options = {
                    "temperature": self.temperature,
                    "repeat_penalty": self.repeat_penalty,
                    "num_predict": self.num_predict,
                }
                if self.stop:
                    options["stop"] = self.stop
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    options=options,
                )
                text = response["message"]["content"]
                decision = self._parse_response(text, context)
                return decision

            except Exception as e:
                logger.warning(f"Ollama attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        # Fallback: return random strategy
        logger.error("All Ollama retries failed — falling back to random")
        return {"strategy": "random", "params": {}}

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        raise NotImplementedError

    def _parse_response(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Extract the first JSON object from a text string.

        Tries a whole-string parse first (handles nested braces, which the
        flat regex below cannot), then falls back to a brace-counting scan
        for a JSON object embedded in surrounding prose/markdown.
        """
        stripped = text.strip()
        for fence in ("```json", "```"):
            if stripped.startswith(fence):
                stripped = stripped[len(fence):]
        stripped = stripped.strip().rstrip("`").strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            start = text.find("{", start + 1)
        return None
