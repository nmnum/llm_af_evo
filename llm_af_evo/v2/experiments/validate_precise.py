"""
One-off: same validation as v2/experiments/validate_population_2b.py but prints
full-precision mean_hv_2b instead of rounding to 1 decimal, to check whether the
4 candidates that displayed as tied (446.9) are actually exactly equal or just close.

Points at the main checkout's llm_af_evo/ for data + code (not this worktree's
copy) since that's where the real evolution run's output and oracle data live.
"""
import json
import pathlib
import sys

import numpy as np
from scipy.stats import wilcoxon

_LLM_AF_EVO = pathlib.Path("/home/nehamungale/ls_na_egbo/llm_af_evo")
for _p in (
    _LLM_AF_EVO.parent,
    _LLM_AF_EVO / "shared",
    _LLM_AF_EVO / "v1_pre_v2" / "src",
    _LLM_AF_EVO / "v1_pre_v2" / "experiments",
    _LLM_AF_EVO / "v2" / "src",
    _LLM_AF_EVO / "v2" / "experiments",
):
    sys.path.insert(0, str(_p))
from full_replay import run_2b_campaign, run_baseline_campaign

POP_PATH = _LLM_AF_EVO / "v2/experiments/evolution_runs/run_v2_mAb_dro_hvi/final_population.json"
HELDOUT_DIR = _LLM_AF_EVO / "data/training_logs_coatings_100/heldout"
ORACLE = "coatings"
N_SEEDS = 1
FITNESS_SEED_STRIDE = 1000

population = json.load(open(POP_PATH))
logs = [json.load(open(f)) for f in sorted(HELDOUT_DIR.glob("*.json"))]

baseline_hvs = []
for i, log in enumerate(logs):
    seeds = [i + r * FITNESS_SEED_STRIDE for r in range(N_SEEDS)]
    repeat_hvs = [run_baseline_campaign(log, seed=s, oracle_family=ORACLE)["final_hv"] for s in seeds]
    baseline_hvs.append(np.mean(repeat_hvs))
baseline_hvs = np.array(baseline_hvs)
print(f"Baseline mean: {baseline_hvs.mean():.10f}\n")

rows = []
for p in population:
    af_hvs = []
    for i, log in enumerate(logs):
        seeds = [i + r * FITNESS_SEED_STRIDE for r in range(N_SEEDS)]
        repeat_hvs = [run_2b_campaign(p["code"], log, seed=s, oracle_family=ORACLE)["final_hv"] for s in seeds]
        af_hvs.append(np.mean(repeat_hvs))
    af_hvs = np.array(af_hvs)
    rows.append({"id": p["id"], "mean_hv_2b": float(af_hvs.mean()), "af_hvs": af_hvs.tolist()})
    print(f"  {p['id']:<16} mean_hv_2b={af_hvs.mean():.10f}")

print("\nPairwise exact-equality check among the 4 previously-tied candidates:")
tied_ids = ["gen16_child2", "gen1_child2", "gen8_child1", "gen20_child3"]
tied_rows = {r["id"]: r for r in rows if r["id"] in tied_ids}
ids = list(tied_rows.keys())
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        a, b = tied_rows[ids[i]], tied_rows[ids[j]]
        exact = a["mean_hv_2b"] == b["mean_hv_2b"]
        print(f"  {ids[i]} vs {ids[j]}: {a['mean_hv_2b']:.10f} vs {b['mean_hv_2b']:.10f}  exact_equal={exact}  diff={a['mean_hv_2b']-b['mean_hv_2b']:.10f}")
