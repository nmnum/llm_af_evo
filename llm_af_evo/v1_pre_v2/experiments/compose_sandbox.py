"""
compose_sandbox.py — subprocess execution sandbox for compose_batch
programs (compose_interface.py's contract). Same isolation pattern as
sandbox.py/gen_sandbox.py (subprocess + argv-JSON + tempfile, enforced
import whitelist, numpy.random pre-warm fix carried over from
gen_sandbox.py's fix), context construction identical to sandbox.py's
(same pool/gp_posterior/pareto_front/campaign shape) plus the batch size
k as an explicit argument.

Validation enforces the duplicate-index guard structurally (Step 2 of the
plan): compose_batch returning a repeated index — the trivial "pick one
high-mu candidate k times" gaming failure mode — is a hard SandboxError,
not something left to 2b replay to catch after the fact.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ALLOWED_IMPORTS = {"numpy", "math", "itertools", "np"}
SANDBOX_TIMEOUT = 10  # seconds


class ComposeSandboxError(Exception):
    pass


_WRAPPER_HEADER = '''import sys, json, builtins, warnings
warnings.filterwarnings("ignore")
import numpy as np
import numpy.random  # noqa: F401 — pre-warm before the whitelist patch below;
# see gen_sandbox.py's identical fix for why (numpy.random's lazy submodule
# load does its own internal `import warnings`, which the restricted
# importer would otherwise block if the first access happens after the
# patch).

_real_import = builtins.__import__
_WHITELIST = %r

def _restricted_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top not in _WHITELIST:
        raise ImportError("import of %%r is not permitted in the compose sandbox" %% name)
    return _real_import(name, *args, **kwargs)

builtins.__import__ = _restricted_import

# ── Candidate compose_batch code ────────────────────────────────────────────
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
    k = json.loads(sys.argv[12])

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

    result = compose_batch(context, k)
    if result is None:
        raise ValueError("compose_batch() returned None — missing return statement")
    result = [int(v) for v in result]
    if len(result) != k:
        raise ValueError(f"compose_batch() returned {len(result)} indices, expected k={k}")
    if len(set(result)) != len(result):
        raise ValueError(f"compose_batch() returned duplicate indices: {result}")
    if any(v < 0 or v >= len(pool_x) for v in result):
        raise ValueError(
            f"compose_batch() returned an out-of-range index (pool size {len(pool_x)}): {result}")
    print(json.dumps(result))
'''


def _build_sandbox_wrapper(compose_code: str) -> str:
    return _WRAPPER_HEADER + compose_code + _WRAPPER_FOOTER


def run_compose_in_sandbox(compose_code: str, pool_x: np.ndarray, pool_mu: np.ndarray,
                            pool_sigma: np.ndarray, X_obs: np.ndarray,
                            front_allmax: np.ndarray, ref_point_allmax: np.ndarray,
                            step: int, budget: int, n_obs: int, stagnant_batches: int,
                            k: int, objective_names=None,
                            log_dir: pathlib.Path = None,
                            fail_log_dir: pathlib.Path = None) -> list:
    """
    Execute compose_code (must define compose_batch per compose_interface.py's
    contract) in a subprocess and return a list of k distinct valid pool
    indices. Raises ComposeSandboxError on timeout, non-zero exit,
    malformed output, wrong length, duplicate indices, or out-of-range
    indices.
    """
    if objective_names is None:
        objective_names = ["Tm", "kD", "viscosity"]
    wrapper = _build_sandbox_wrapper(compose_code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        tmp_path = f.name

    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(log_dir.glob("call_*.py")))
        (log_dir / f"call_{existing:05d}.py").write_text(compose_code)

    try:
        result = subprocess.run(
            [sys.executable, tmp_path,
             json.dumps(pool_x.tolist()), json.dumps(pool_mu.tolist()),
             json.dumps(pool_sigma.tolist()), json.dumps(X_obs.tolist()),
             json.dumps(front_allmax.tolist()), json.dumps(ref_point_allmax.tolist()),
             json.dumps(int(step)), json.dumps(int(budget)),
             json.dumps(int(n_obs)), json.dumps(int(stagnant_batches)),
             json.dumps(list(objective_names)), json.dumps(int(k))],
            capture_output=True, text=True, timeout=SANDBOX_TIMEOUT,
        )

        if result.returncode != 0:
            if fail_log_dir is not None:
                fail_log_dir = pathlib.Path(fail_log_dir)
                fail_log_dir.mkdir(parents=True, exist_ok=True)
                (fail_log_dir / "last_failed_compose_code.py").write_text(compose_code)
                (fail_log_dir / "last_failed_compose_stderr.txt").write_text(result.stderr)
            raise ComposeSandboxError(
                f"compose_batch program exited nonzero: {result.stderr[-1000:]}")

        stdout = result.stdout.strip()
        if not stdout:
            raise ComposeSandboxError(
                f"compose_batch program produced no stdout (stderr: {result.stderr[:200]})")

        try:
            indices = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise ComposeSandboxError(f"compose_batch program's stdout was not valid JSON: {e}")

    except subprocess.TimeoutExpired:
        raise ComposeSandboxError(f"compose_batch program timed out after {SANDBOX_TIMEOUT}s")
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    n = len(pool_x)
    if len(indices) != k:
        raise ComposeSandboxError(f"compose_batch program returned {len(indices)} indices, expected {k}")
    if len(set(indices)) != len(indices):
        raise ComposeSandboxError(f"compose_batch program returned duplicate indices: {indices}")
    if any(v < 0 or v >= n for v in indices):
        raise ComposeSandboxError(f"compose_batch program returned out-of-range index in {indices} (pool size {n})")

    return list(indices)
