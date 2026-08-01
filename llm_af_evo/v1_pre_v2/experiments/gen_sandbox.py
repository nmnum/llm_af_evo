"""
gen_sandbox.py — subprocess execution sandbox for LLM-evolved candidate-
GENERATION programs (propose_candidates, per gen_interface.py's contract).
Same isolation pattern as sandbox.py (subprocess + argv-JSON + tempfile,
same timeout/failure handling, same enforced import whitelist) — a
separate module rather than extending sandbox.py because the I/O shape and
context contents are structurally different (no GP posterior in, an
(n_candidates, n_dims) array out instead of an (n_pool,) score vector).

This is Gate 1 of a two-gate design (see gen_interface.py's module
docstring): geometry-only proposers here, no GP/acquisition access at all.
A hypothetical Gate 2 (GP-proxy inside the sandbox, giving a proposer
gradient-informed access to posterior means/stds at arbitrary x) is
explicitly NOT built here — only worth building if Gate 1 is null, and even
then any win there is harder to attribute to "LLM generation logic" vs.
"another acquisition optimizer," per the plan's own caveat.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ALLOWED_IMPORTS = {"numpy", "math", "itertools", "np"}
SANDBOX_TIMEOUT = 10  # seconds


class GenSandboxError(Exception):
    pass


_WRAPPER_HEADER = '''import sys, json, builtins, warnings
warnings.filterwarnings("ignore")
import numpy as np
import numpy.random  # noqa: F401 — pre-warm BEFORE the import whitelist below
# is installed. numpy.random is lazily submodule-loaded on first attribute
# access (e.g. the first np.random.default_rng(...) call in candidate code),
# and that lazy load internally does its own `import warnings` — nothing to
# do with candidate code importing warnings itself. If that first access
# happens after builtins.__import__ is patched, the restricted importer
# blocks numpy's OWN internal import and every candidate using
# np.random.default_rng fails with a misleading "import of warnings is not
# permitted" error. Touching numpy.random here, before the patch, forces
# that one-time internal import to happen while imports are unrestricted.

_real_import = builtins.__import__
_WHITELIST = %r

def _restricted_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top not in _WHITELIST:
        raise ImportError("import of %%r is not permitted in the generation sandbox" %% name)
    return _real_import(name, *args, **kwargs)

builtins.__import__ = _restricted_import

# ── Candidate propose_candidates code ──────────────────────────────────────
''' % (ALLOWED_IMPORTS,)

_WRAPPER_FOOTER = '''
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_obs = np.array(json.loads(sys.argv[1]))
    pareto_x = np.array(json.loads(sys.argv[2]))
    step = json.loads(sys.argv[3])
    budget = json.loads(sys.argv[4])
    n_obs = json.loads(sys.argv[5])
    stagnant_batches = json.loads(sys.argv[6])
    n_dims = json.loads(sys.argv[7])
    n_candidates = json.loads(sys.argv[8])

    context = {
        "X_obs": X_obs,
        "pareto_x": pareto_x,
        "campaign": {
            "step": step, "budget": budget, "progress": step / max(budget, 1),
            "n_obs": n_obs, "stagnant_batches": stagnant_batches,
        },
        "n_dims": n_dims,
        "n_candidates": n_candidates,
    }

    result = propose_candidates(context)
    if result is None:
        raise ValueError("propose_candidates() returned None — missing return statement")
    result = np.asarray(result, dtype=float)
    if result.shape != (n_candidates, n_dims):
        raise ValueError(
            f"propose_candidates() returned shape {result.shape}, expected "
            f"({n_candidates}, {n_dims})")
    print(json.dumps(result.tolist()))
'''


def _build_sandbox_wrapper(gen_code: str) -> str:
    # Plain concatenation, not an f-string over the whole template — same
    # reasoning as sandbox.py's _build_sandbox_wrapper (gen_code may itself
    # contain `{`/`}`).
    return _WRAPPER_HEADER + gen_code + _WRAPPER_FOOTER


def run_generator_in_sandbox(gen_code: str, X_obs: np.ndarray, pareto_x: np.ndarray,
                              step: int, budget: int, n_obs: int, stagnant_batches: int,
                              n_dims: int, n_candidates: int,
                              log_dir: pathlib.Path = None,
                              fail_log_dir: pathlib.Path = None) -> np.ndarray:
    """
    Execute gen_code (must define propose_candidates per gen_interface.py's
    contract) in a subprocess and return the resulting (n_candidates,
    n_dims) array, clipped to [0, 1]. Raises GenSandboxError on timeout,
    non-zero exit, malformed output, wrong shape, or non-finite values.
    """
    wrapper = _build_sandbox_wrapper(gen_code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        tmp_path = f.name

    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(log_dir.glob("call_*.py")))
        (log_dir / f"call_{existing:05d}.py").write_text(gen_code)

    try:
        result = subprocess.run(
            [sys.executable, tmp_path,
             json.dumps(X_obs.tolist()), json.dumps(pareto_x.tolist()),
             json.dumps(int(step)), json.dumps(int(budget)),
             json.dumps(int(n_obs)), json.dumps(int(stagnant_batches)),
             json.dumps(int(n_dims)), json.dumps(int(n_candidates))],
            capture_output=True, text=True, timeout=SANDBOX_TIMEOUT,
        )

        if result.returncode != 0:
            if fail_log_dir is not None:
                fail_log_dir = pathlib.Path(fail_log_dir)
                fail_log_dir.mkdir(parents=True, exist_ok=True)
                (fail_log_dir / "last_failed_gen_code.py").write_text(gen_code)
                (fail_log_dir / "last_failed_gen_stderr.txt").write_text(result.stderr)
            raise GenSandboxError(f"generator program exited nonzero: {result.stderr[-1000:]}")

        stdout = result.stdout.strip()
        if not stdout:
            raise GenSandboxError(
                f"generator program produced no stdout (stderr: {result.stderr[:200]})")

        try:
            candidates = np.array(json.loads(stdout))
        except json.JSONDecodeError as e:
            raise GenSandboxError(f"generator program's stdout was not valid JSON: {e}")

    except subprocess.TimeoutExpired:
        raise GenSandboxError(f"generator program timed out after {SANDBOX_TIMEOUT}s")
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    if candidates.shape != (n_candidates, n_dims):
        raise GenSandboxError(
            f"generator program returned shape {candidates.shape}, "
            f"expected ({n_candidates}, {n_dims})")
    if not np.all(np.isfinite(candidates)):
        raise GenSandboxError("generator program returned non-finite candidates")

    return np.clip(candidates, 0.0, 1.0)
