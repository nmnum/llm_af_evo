"""
sandbox.py — subprocess execution sandbox for LLM-evolved AF (acquisition
function) programs.

Adapted from sdl-adaptive's controllers/approach_c.py (the LLM-rewrites-the-
optimiser controller) — same subprocess-isolation + argv-JSON + tempfile
pattern (_build_sandbox_wrapper / _run_in_sandbox), same failure handling
(timeout -> warn, non-zero exit -> save failing code + stderr, wrong shape
or non-finite output -> reject), same "10 seconds, no network, no file
writes" safety envelope. Adapted here for score_pool()'s I/O shape (a score
vector over a whole candidate pool, per af_interface.py's contract) instead
of approach_c's suggest() (a single next point).

One deliberate addition beyond the original: approach_c.py DECLARES
ALLOWED_IMPORTS but never actually enforces it at runtime — the whitelist
only appears as text in the LLM's prompt ("Allowed imports: numpy, scipy,
sklearn, math, random."), which restrains a cooperative LLM but does
nothing against a program that imports something else anyway. This version
enforces the whitelist inside the subprocess via a restricted __import__,
so a violation is an actual sandbox rejection, not a prompting convention.
The whitelist itself is also narrower here (numpy, math, itertools — no
scipy/sklearn) by design: L1 evolves closed-form scoring expressions, not
model-fitting routines, which is approach K's territory, not this one's.
"""

import json
import logging
import pathlib
import subprocess
import sys
import tempfile
import textwrap

import numpy as np

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
):
    sys.path.insert(0, str(_p))

from af_interface import extract_af_docstring

logger = logging.getLogger(__name__)

ALLOWED_IMPORTS = {"numpy", "math", "itertools", "np"}
SANDBOX_TIMEOUT = 10  # seconds


class SandboxError(Exception):
    pass


_WRAPPER_HEADER = '''import sys, json, builtins, warnings
warnings.filterwarnings("ignore")
import numpy as np
import numpy.random  # noqa: F401 — pre-warm before the import whitelist is
# installed below, same fix as gen_sandbox.py's identical issue: numpy.random
# is lazily submodule-loaded on first access and that lazy load does its own
# internal `import warnings`, which the restricted importer would otherwise
# block if the first access happens after the patch (e.g. any evolved
# score_pool that calls np.random.* itself, even though no current seed
# program does).

_real_import = builtins.__import__
_WHITELIST = %r

def _restricted_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top not in _WHITELIST:
        raise ImportError("import of %%r is not permitted in the AF sandbox" %% name)
    return _real_import(name, *args, **kwargs)

builtins.__import__ = _restricted_import

# ── Candidate score_pool code ──────────────────────────────────────────────
''' % (ALLOWED_IMPORTS,)

_WRAPPER_FOOTER = '''
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pool_x = np.array(json.loads(sys.argv[1]))
    pool_mu = np.array(json.loads(sys.argv[2]))
    pool_sigma = np.array(json.loads(sys.argv[3]))
    X_obs = np.array(json.loads(sys.argv[4]))
    front_allmax = np.array(json.loads(sys.argv[5]))
    ref_point_allmax = np.array(json.loads(sys.argv[6]))
    step = json.loads(sys.argv[7])
    budget = json.loads(sys.argv[8])
    n_obs = json.loads(sys.argv[9])
    stagnant_batches = json.loads(sys.argv[10])
    OBJECTIVE_NAMES = json.loads(sys.argv[11])
    Y_obs = np.array(json.loads(sys.argv[12]))

    # Translate flat, positional arrays into the self-documenting nested
    # context dict per af_interface.py's contract, BEFORE candidate code
    # ever runs — this translation is trusted runner code (not sandboxed
    # exec'd code), so it can use whatever it needs regardless of the
    # import whitelist below. Doing the translation here (not in the
    # caller) keeps run_af_in_sandbox's own signature — and every caller
    # of it — unchanged.

    pool = []
    for i in range(len(pool_x)):
        gp = {name: {"mean": float(pool_mu[i, j]), "std": float(pool_sigma[i, j])}
              for j, name in enumerate(OBJECTIVE_NAMES)}
        pool.append({"x": pool_x[i], "gp_posterior": gp})

    if len(front_allmax) > 0:
        pareto_front_range = {
            name: max(float(front_allmax[:, j].max() - front_allmax[:, j].min()), 1e-6)
            for j, name in enumerate(OBJECTIVE_NAMES)
        }
    else:
        pareto_front_range = {name: 1e-6 for name in OBJECTIVE_NAMES}

    context = {
        "pool": pool,
        "X_obs": X_obs,
        "Y_obs": Y_obs,
        "objective_names": OBJECTIVE_NAMES,
        "pareto_front": front_allmax,
        "pareto_front_range": pareto_front_range,
        "ref_point": ref_point_allmax,
        "ref_point_by_name": {name: float(ref_point_allmax[j])
                               for j, name in enumerate(OBJECTIVE_NAMES)},
        "campaign": {
            "step": step, "budget": budget, "progress": step / max(budget, 1),
            "n_obs": n_obs, "stagnant_batches": stagnant_batches,
        },
    }

    result = score_pool(context)
    if result is None:
        raise ValueError("score_pool() returned None — missing return statement")
    result = list(result)
    if len(result) != len(pool_x):
        raise ValueError(
            f"score_pool() returned {len(result)} scores but there were "
            f"{len(pool_x)} candidates in the pool — lengths must match")
    print(json.dumps([float(v) for v in result]))
'''


def _build_sandbox_wrapper(af_code: str) -> str:
    """
    Wrap the candidate score_pool code in a subprocess-safe script that
    reads its arrays from argv (JSON) and prints the result as JSON.
    Deliberately NOT an f-string over the whole template — af_code is
    arbitrary (possibly LLM-generated) Python and may itself contain `{`
    or `}` (dict/set literals, nested f-strings), which would corrupt an
    f-string interpolation silently or raise a spurious format error.
    Plain concatenation keeps af_code's own braces inert.
    """
    return _WRAPPER_HEADER + af_code + _WRAPPER_FOOTER


def run_af_in_sandbox(af_code: str, pool_x: np.ndarray, pool_mu: np.ndarray,
                       pool_sigma: np.ndarray, X_obs: np.ndarray,
                       front_allmax: np.ndarray, ref_point_allmax: np.ndarray,
                       step: int, budget: int, n_obs: int, stagnant_batches: int,
                       Y_obs: np.ndarray,
                       objective_names=None,
                       log_dir: pathlib.Path = None,
                       fail_log_dir: pathlib.Path = None) -> np.ndarray:
    """
    Execute af_code (must define score_pool per af_interface.py's contract)
    in a subprocess and return the resulting (N,) score array. Raises
    SandboxError on timeout, non-zero exit, malformed output, wrong shape,
    or non-finite values — callers are expected to fall back to the last
    known-good AF, exactly as approach_c.py's _parse_response does.

    objective_names defaults to ["Tm", "kD", "viscosity"] for backward
    compatibility with every existing excipient call site — pass an
    oracle's own .objective_names() explicitly when running on a
    different oracle (e.g. ada_coatings_oracle's conductance/
    xrf_conductance/conductivity), same pattern as
    excipient_campaign_mo.pareto_front_of's directions parameter. Note
    pool_mu/pool_sigma are ALREADY in all-maximise convention by the time
    they reach here (upstream strategy code did the sign flip) — AF code
    reading context["gp_posterior"][name]["mean"] never needs to re-apply
    objective_directions itself, only objective_names for which keys exist.

    Y_obs is every observed outcome so far (not just the non-dominated
    front), already in all-maximise convention like pool_mu/front_allmax —
    pass front_allmax itself if the caller doesn't separately track a
    filtered front, since both are derived from the same running Y array.
    """
    if objective_names is None:
        objective_names = ["Tm", "kD", "viscosity"]

    # Explainability-line enforcement: reject before ever spawning the
    # subprocess (no point burning the 10s timeout budget on code that's
    # already invalid). Only the docstring's first line is required to be
    # non-empty — a docstring may continue past it, but the first line is
    # the one contract with any tooling: it's what makes an evolved AF's
    # results-table row and the next generation's crossover prompt
    # human-readable (see evolve_af.py's module docstring).
    if not extract_af_docstring(af_code):
        raise SandboxError(
            "score_pool is missing a required one-line docstring as its "
            "first statement (e.g. def score_pool(context):\\n    \"\"\"<what "
            "this AF does, in plain English>\"\"\")")

    n = len(pool_x)
    wrapper = _build_sandbox_wrapper(af_code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        tmp_path = f.name

    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(log_dir.glob("call_*.py")))
        (log_dir / f"call_{existing:05d}.py").write_text(af_code)

    try:
        result = subprocess.run(
            [sys.executable, tmp_path,
             json.dumps(pool_x.tolist()), json.dumps(pool_mu.tolist()),
             json.dumps(pool_sigma.tolist()), json.dumps(X_obs.tolist()),
             json.dumps(front_allmax.tolist()), json.dumps(ref_point_allmax.tolist()),
             json.dumps(int(step)), json.dumps(int(budget)),
             json.dumps(int(n_obs)), json.dumps(int(stagnant_batches)),
             json.dumps(list(objective_names)), json.dumps(Y_obs.tolist())],
            capture_output=True, text=True, timeout=SANDBOX_TIMEOUT,
        )

        if result.returncode != 0:
            if fail_log_dir is not None:
                fail_log_dir = pathlib.Path(fail_log_dir)
                fail_log_dir.mkdir(parents=True, exist_ok=True)
                (fail_log_dir / "last_failed_code.py").write_text(af_code)
                (fail_log_dir / "last_failed_stderr.txt").write_text(result.stderr)
            raise SandboxError(f"AF program exited nonzero: {result.stderr[-1000:]}")

        stdout = result.stdout.strip()
        if not stdout:
            raise SandboxError(f"AF program produced no stdout (stderr: {result.stderr[:200]})")

        try:
            scores = np.array(json.loads(stdout))
        except json.JSONDecodeError as e:
            raise SandboxError(f"AF program's stdout was not valid JSON: {e}")

    except subprocess.TimeoutExpired:
        raise SandboxError(f"AF program timed out after {SANDBOX_TIMEOUT}s")
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    if scores.shape != (n,):
        raise SandboxError(f"AF program returned shape {scores.shape}, expected ({n},)")
    if not np.all(np.isfinite(scores)):
        raise SandboxError("AF program returned non-finite scores")

    return scores
