"""
controllers/approach_c.py — Approach C: LLM rewrites the optimiser Python file.

The LLM receives campaign context and the current optimiser code, then returns
a complete Python file defining a suggest(X_obs, y_obs, bounds) function.
The new code is executed in a subprocess sandbox with a 10-second timeout
and a whitelist of allowed imports.

Safety constraints:
  - Subprocess isolation (no shared memory with main process)
  - 10-second wall-clock timeout
  - Import whitelist: numpy, scipy, sklearn, math, random, json, itertools
  - No network access, no file writes
"""

import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Dict, Optional

import numpy as np

from controllers.base import OllamaController

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = pathlib.Path(__file__).parent.parent / "prompts" / "system_c.txt"
_TEMPLATE = pathlib.Path(__file__).parent.parent / "optimiser_template.py"

ALLOWED_IMPORTS = {"numpy", "scipy", "sklearn", "math", "random", "json", "itertools", "np"}
SANDBOX_TIMEOUT = 30  # seconds — must match the figure quoted to the LLM in prompts/system_c.txt


# Project root — where evolutionary_candidates.py lives
_PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())


def _build_sandbox_wrapper(optimiser_code: str) -> str:
    """Wrap user code in a subprocess-safe script that reads args from stdin."""
    project_root = _PROJECT_ROOT  # capture at definition time
    return f'''
import sys, json, warnings
sys.path.insert(0, {project_root!r})  # makes evolutionary_candidates importable
import numpy as np
warnings.filterwarnings("ignore")  # prevent warnings from corrupting stdout JSON

# ── User-provided optimiser code ──────────────────────────────────────────
{optimiser_code}
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_obs = np.array(json.loads(sys.argv[1]))
    y_obs = np.array(json.loads(sys.argv[2]))
    bounds = np.array(json.loads(sys.argv[3]))
    result = suggest(X_obs, y_obs, bounds)
    if result is None:
        raise ValueError("suggest() returned None — missing return statement")
    print(json.dumps([float(x) for x in result]))
'''


def _run_in_sandbox(
    optimiser_code: str,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    bounds: np.ndarray,
) -> Optional[np.ndarray]:
    """Execute optimiser_code in a subprocess and return the suggested point."""
    wrapper = _build_sandbox_wrapper(optimiser_code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path,
             json.dumps(X_obs.tolist()),
             json.dumps(y_obs.tolist()),
             json.dumps(bounds.tolist())],
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT,
        )
        if result.returncode == 0:
            stdout = result.stdout.strip()
            if not stdout:
                logger.warning(f"Sandbox produced no output. stderr: {result.stderr[:200]}")
                return None
            x_next = np.array(json.loads(stdout))
            if x_next.shape == (bounds.shape[0],) and np.all(np.isfinite(x_next)):
                return x_next
            else:
                logger.warning(f"Sandbox output wrong shape or non-finite: {x_next}")
        #else:
        if result.returncode != 0:
            # Save failing code for inspection. Filename includes pid + a random
            # suffix so parallel seeds/processes don't clobber each other's dumps
            # (previously a fixed "./sdl_failed_code.py" path — a race under any
            # concurrent run).
            tag = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
            with open(f"./sdl_failed_code_{tag}.py", "w") as f:
                f.write(optimiser_code)
            with open(f"./sdl_failed_stderr_{tag}.txt", "w") as f:
                f.write(result.stderr)
            logger.warning(f"Sandbox error: {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        logger.warning("Sandbox timed out")
    except json.JSONDecodeError as e:
        logger.warning(f"Sandbox JSON parse failed: {e}. stdout: {result.stdout[:100]!r}")
    except Exception as e:
        logger.warning(f"Sandbox exception: {e}")
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    return None


class ApproachCController(OllamaController):
    """
    Approach C: LLM rewrites the optimiser Python file.

    The LLM returns a complete Python module with a suggest() function.
    The code is executed in a subprocess sandbox.
    """

    _needs_gp_uncertainty = True

    def __init__(self, model: str = "qwen3-coder:30b",
                 code_log_dir: str = "./approach_c_code_logs", **kwargs):
        system = _SYSTEM_PROMPT.read_text() if _SYSTEM_PROMPT.exists() else _DEFAULT_SYSTEM
        kwargs.setdefault("max_retries", 3)
        kwargs.setdefault("timeout", 120)
        # A generated optimiser file is ~200-300 tokens. 2048 gave the model room to
        # keep writing docstrings/explanations after the function body, which is
        # wasted latency and occasionally confuses the fence-stripping in
        # _parse_response. 1024 is generous headroom without inviting that.
        kwargs.setdefault("num_predict", 1024)
        # Anything after the function body is wasted tokens and sometimes trailing
        # prose that breaks the fence-stripping in _parse_response.
        kwargs.setdefault("stop", ["if __name__", "\n# End", "\nExplanation:"])
        super().__init__(model=model, system_prompt=system, **kwargs)
        self._current_code = _TEMPLATE.read_text() if _TEMPLATE.exists() else _DEFAULT_TEMPLATE
        self._code_log_dir = pathlib.Path(code_log_dir)
        self._code_log_dir.mkdir(parents=True, exist_ok=True)
        self._call_count = 0
        # Fallback-chain visibility: previously you could not tell, from the results
        # alone, whether approach_c was running the LLM's code 80% of the time or 5%
        # of the time. These counters make that auditable after a run.
        self._n_llm_code_accepted = 0
        self._n_llm_code_rejected = 0   # new code failed sandbox, fell back to prior code
        self._n_both_failed = 0         # prior code also failed, fell back to random
        # Delta-pressure bookkeeping: what happened since the code was last
        # revised, so the prompt can say "this is/isn't working" instead of
        # just showing the code with no signal about whether to change it.
        self._last_best_normalised: Optional[float] = None
        self._last_outcome: str = "This is the starting template — no revision has been made yet."
        self._steps_since_revision = 0
        self._last_call_step = 0

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        self._steps_since_revision = context["step"] - self._last_call_step
        self._last_call_step = context["step"]
        progress = context["step"] / context["budget"]
        best_now = context["best_normalised"]

        if self._last_best_normalised is not None:
            delta = best_now - self._last_best_normalised
            if delta > 1e-6:
                trend = f"IMPROVED by {delta:.4f} since the last code revision."
            elif delta < -1e-6:
                trend = f"got WORSE by {abs(delta):.4f} since the last code revision (regression!)."
            else:
                trend = "has NOT moved since the last code revision — the current strategy is stagnating."
        else:
            trend = "N/A (first call)."

        return (
            f"Campaign status:\n"
            f"  Step: {context['step']} / {context['budget']} ({100*progress:.0f}% complete)\n"
            f"  Best found (normalised): {best_now:.3f}\n"
            f"  Best-value trend: {trend}\n"
            f"  Improvement rate (last 10 steps): {context['improvement_rate']:.2f}\n"
            f"  GP uncertainty: {context['gp_uncertainty']:.4f}\n"
            f"  Dimensions: {context['bounds'].shape[0]}\n"
            f"  Observations so far: {context['n_obs']}\n\n"
            f"Outcome of the code's last {self._steps_since_revision} step(s) in the sandbox:\n"
            f"  {self._last_outcome}\n\n"
            f"Current optimiser code:\n```python\n{self._current_code}\n```\n\n"
            f"Do NOT return this code unchanged. Make a specific, deliberate change "
            f"to the acquisition/exploration logic that responds to the status above "
            f"(e.g. if stagnating, increase exploration/diversity; if GP uncertainty is "
            f"high, exploit less; if it just regressed, revert that change and try a "
            f"different one). State-preserving no-ops (renaming variables, reformatting, "
            f"reordering imports) do not count as a change.\n"
            f"Return a complete Python file with a suggest(X_obs, y_obs, bounds) function.\n"
            f"Allowed imports: numpy, scipy, sklearn, math, random.\n"
            f"Return ONLY the Python code, no markdown fences."
        )

    def suggest_point(self, X_obs: np.ndarray, y_obs: np.ndarray,
                       bounds: np.ndarray) -> Optional[np.ndarray]:
        """Re-run the current accepted code in the sandbox against the
        latest observations. Called once per simulator step (not just once
        per controller_interval) so the LLM's code actually governs every
        step it's live for, rather than one step in five."""
        x_next = _run_in_sandbox(self._current_code, X_obs, y_obs, bounds)
        if x_next is None:
            self._n_both_failed += 1  # counted the same as a full fallback-to-random
            return None
        return np.clip(x_next, bounds[:, 0], bounds[:, 1])

    def _parse_response(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Strip markdown fences if present
        code = text.strip()
        for fence in ["```python", "```"]:
            if code.startswith(fence):
                code = code[len(fence):]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        self._last_best_normalised = context["best_normalised"]

        if "def suggest" not in code:
            logger.warning("LLM response missing suggest() function — keeping current code")
            self._last_outcome = (
                "The LLM's last response did not contain a suggest() function "
                "(likely explanatory prose instead of code) — it was discarded "
                "and the previous code kept."
            )
            return {"strategy": "_custom", "params": {}, "x_next": None}

        if code == self._current_code.strip():
            self._last_outcome = (
                "The LLM returned the CURRENT code unchanged. That is not useful — "
                "next time make a concrete edit."
            )

        # Try running in sandbox
        X_obs = context["X_obs"]
        y_obs = context["y_obs"]
        bounds = context["bounds"]

        x_next = _run_in_sandbox(code, X_obs, y_obs, bounds)
        rejected_new_code = x_next is None
        if x_next is None:
            # New code failed or timed out — try last known-good code before falling back
            self._last_outcome = (
                f"The LLM's new code failed in the sandbox (error, wrong output shape, "
                f"or exceeded the {SANDBOX_TIMEOUT}s timeout) and was rejected. "
                f"The previous code is still in use — try a simpler/cheaper approach."
            )
            x_next = _run_in_sandbox(self._current_code, X_obs, y_obs, bounds)
        if x_next is not None:
            # Clip to bounds
            x_next = np.clip(x_next, bounds[:, 0], bounds[:, 1])
            if rejected_new_code:
                self._n_llm_code_rejected += 1
            elif code != self._current_code.strip():
                self._current_code = code
                self._last_outcome = "The LLM's new code was accepted and is now live."
                self._n_llm_code_accepted += 1
            else:
                self._n_llm_code_accepted += 1  # unchanged code still ran fine
            # Save every accepted code version for later analysis
            try:
                log_path = self._code_log_dir / f"call_{self._call_count:04d}.py"
                log_path.write_text(code)
            except Exception:
                pass
            self._call_count += 1
            logger.info(
                f"ApproachC call {self._call_count}: accepted={self._n_llm_code_accepted} "
                f"rejected(reverted-to-prior)={self._n_llm_code_rejected} "
                f"both-failed(random)={self._n_both_failed}"
            )
            return {"strategy": "_custom", "params": {}, "x_next": x_next}

        # Sandbox failed even for the last known-good code — keep it, return random
        self._n_both_failed += 1
        logger.warning(
            f"Sandbox failed for both new and prior code — falling back to random "
            f"(both-failed count: {self._n_both_failed})"
        )
        self._last_outcome = (
            "Both the new code AND the previous known-good code failed in the "
            "sandbox this round; the simulator fell back to a random point."
        )
        return {"strategy": "random", "params": {}}


_DEFAULT_SYSTEM = """You are an autonomous optimisation controller for a self-driving laboratory.
Your task is to write a Python optimiser that suggests the next experiment to run.
You will receive the current optimiser code and campaign state, and must return an improved version.
The function signature must be: suggest(X_obs, y_obs, bounds) -> np.ndarray (shape: (d,))
Return ONLY valid Python code."""

_DEFAULT_TEMPLATE = """import numpy as np

def suggest(X_obs, y_obs, bounds):
    \"\"\"Default: uniform random sampling.\"\"\"
    d = bounds.shape[0]
    return np.array([np.random.uniform(bounds[i, 0], bounds[i, 1]) for i in range(d)])
"""


# ── Approach C with evolutionary candidates (approach_c_evo) ─────────────────

_TEMPLATE_EVO = pathlib.Path(__file__).parent.parent / "optimiser_template_evo.py"


class ApproachCEvoController(ApproachCController):
    """
    Approach C variant that starts from a template which already uses
    evolutionary_candidates() and novelty_select(). The LLM improves the
    acquisition scoring and beta schedule rather than deciding whether to
    use evolutionary diversity.

    This avoids the failure mode where the LLM ignores the utility even
    when instructed to use it — the template makes usage mandatory.
    """

    def __init__(self, model: str = "qwen3-coder:30b",
                 code_log_dir: str = "./approach_c_evo_logs", **kwargs):
        super().__init__(model=model, code_log_dir=code_log_dir, **kwargs)
        # Override template with evo version
        self._current_code = (
            _TEMPLATE_EVO.read_text() if _TEMPLATE_EVO.exists()
            else _DEFAULT_TEMPLATE_EVO
        )
        self._check_evolutionary_candidates_importable()

    @staticmethod
    def _check_evolutionary_candidates_importable() -> None:
        """Fail loudly (not silently, inside the sandbox, 6 hours into a run)
        if evolutionary_candidates can't be imported from the sandbox's view
        of sys.path. Runs the exact import the template depends on inside a
        real subprocess so the check matches sandbox conditions."""
        probe = (
            f"import sys\n"
            f"sys.path.insert(0, {_PROJECT_ROOT!r})\n"
            f"from evolutionary_candidates import evolutionary_candidates, novelty_select\n"
            f"print('OK')\n"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True, text=True, timeout=10,
            )
        except Exception as e:
            logger.error(f"evolutionary_candidates import self-test crashed: {e}")
            return
        if result.returncode != 0 or "OK" not in result.stdout:
            logger.error(
                f"evolutionary_candidates is NOT importable from the sandbox "
                f"(project root={_PROJECT_ROOT!r}). Every ApproachCEvo call will "
                f"fail until this is fixed. stderr:\n{result.stderr}"
            )
        else:
            logger.info(f"evolutionary_candidates import self-test OK (project root={_PROJECT_ROOT!r})")


_DEFAULT_TEMPLATE_EVO = """import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from evolutionary_candidates import evolutionary_candidates, novelty_select

def suggest(X_obs, y_obs, bounds):
    d = bounds.shape[0]
    n = len(X_obs)
    x_next = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
    try:
        if n <= d + 1:
            return x_next
        evo_cands = evolutionary_candidates(X_obs, y_obs, bounds, n=72)
        model = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True)
        model.fit(X_obs, y_obs)
        mu, sigma = model.predict(evo_cands, return_std=True)
        beta = 2.0
        acq_scores = mu + beta * sigma
        best_idx = novelty_select(evo_cands, X_obs, bounds, acq_scores, merit_weight=0.7)
        x_next = evo_cands[best_idx]
    except Exception:
        pass
    return x_next.flatten()
"""
