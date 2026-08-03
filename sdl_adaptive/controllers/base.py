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
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.timeout = timeout

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call Ollama and return a decision dict."""
        import ollama

        prompt = self._build_prompt(context)

        for attempt in range(self.max_retries):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.1, "num_predict": 2048},
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
        """Extract the first JSON object from a text string."""
        import re
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None
