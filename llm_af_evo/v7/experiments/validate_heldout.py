"""
validate_heldout.py — the v7 stop-criterion check (see v7/README.md's
"Stop criterion"): validates a champion AF against every held-out set —
the 3 training domains' own held-out campaigns (never used for fitness,
even under resampling — resampling only ever draws from train/, not
heldout/) plus the real target, DTLZ2(n_obj=5)@35%CV, never trained on at
all.

Run this against <run_dir>/best_af_fullpool.py (from
select_full_pool_champion.py), NOT <run_dir>/best_af.py — see that
script's docstring for why the raw best_af.py from a v7 run isn't the
right candidate to validate.

Usage:
    python validate_heldout.py --af_path evolution_runs/run1/best_af_fullpool.py
"""

import argparse
import json
import pathlib
import sys

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
    _LLM_AF_EVO / "v6" / "src",
    _LLM_AF_EVO / "v7" / "src",
):
    sys.path.insert(0, str(_p))

import full_replay
from evolve_af_v7 import evaluate_af_2b, compute_baseline_hvs

_V6_EXPERIMENTS = _LLM_AF_EVO / "v6" / "experiments"

DOMAINS = {
    "zdt1_heldout": ("zdt1", _V6_EXPERIMENTS / "training_logs_zdt1" / "heldout"),
    "dtlz2_3obj_heldout": ("dtlz2", _V6_EXPERIMENTS / "training_logs_dtlz2_3obj" / "heldout"),
    "zdt3_heldout": ("zdt3", _V6_EXPERIMENTS / "training_logs_zdt3" / "heldout"),
    "dtlz2_5obj_heldout": ("dtlz2", _V6_EXPERIMENTS / "training_logs_dtlz2_5obj_heldout" / "heldout"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af_path", required=True)
    args = ap.parse_args()

    code = pathlib.Path(args.af_path).read_text()

    for name, (family, dirpath) in DOMAINS.items():
        logs = [json.loads(f.read_text()) for f in sorted(dirpath.glob("*.json"))]
        if name == "dtlz2_5obj_heldout":
            # feature_dim = n_obj - 1 + k = 5 - 1 + 4 = 8; n_objectives = 5.
            # _ORACLE_SHAPE_EXPECTATIONS["dtlz2"] hardcodes the 3-obj training
            # shape (6, 3) — override just for this held-out-only call.
            full_replay._ORACLE_SHAPE_EXPECTATIONS["dtlz2"] = {"feature_dim": 8, "n_objectives": 5}
        print(f"\n=== {name}: {len(logs)} held-out campaigns (oracle_family={family}) ===")
        baseline_hvs = compute_baseline_hvs(logs, oracle_family=family)
        result = evaluate_af_2b(code, logs, baseline_hvs, oracle_family=family)
        print(f"{name}: mean_margin={result['mean_margin']:+.4f} "
              f"median_margin={result['median_margin']:+.4f} "
              f"win_rate={result['win_rate']:.2f} ci_lower_16={result['ci_lower_16']:+.4f} "
              f"se={result.get('se_mean_margin'):.4f}")


if __name__ == "__main__":
    main()
