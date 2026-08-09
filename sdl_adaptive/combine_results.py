"""Assemble a combined results dir from whatever out_dirs have finished so far,
symlinking each dataset/condition folder in place and rebuilding metrics_summary.csv
from the metrics_per_seed.json files already on disk (no re-running needed).
Conditions/datasets that haven't finished yet (approach_c on the 4 pareto_* sets)
are simply absent -- left blank until that process catches up.
"""
import json, pathlib, shutil
import pandas as pd
from oracle import NNOracle
from run_experiment import ALL_DATASETS

OUT_DIRS = ["results_baselines", "results_a", "results_b", "results_c", "results_d_fixed", "results_egbo"]
DATA_DIR = pathlib.Path("data") if pathlib.Path("data").exists() else pathlib.Path(".")
COMBINED = pathlib.Path("results_combined")

if COMBINED.exists():
    shutil.rmtree(COMBINED)
COMBINED.mkdir()

rows = []
gb_cache, gmin_cache = {}, {}

for ds_label, ds_name in ALL_DATASETS:
    for out_dir_name in OUT_DIRS:
        ds_dir = pathlib.Path(out_dir_name) / ds_label
        if not ds_dir.exists():
            continue
        for cond_dir in ds_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            per_seed_path = cond_dir / "metrics_per_seed.json"
            if not per_seed_path.exists():
                continue
            condition = cond_dir.name
            dest = COMBINED / ds_label / condition
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() or dest.is_symlink():
                continue
            dest.symlink_to(cond_dir.resolve())

            if ds_label not in gb_cache:
                oracle = NNOracle.from_dataset(ds_name, str(DATA_DIR))
                gb_cache[ds_label] = oracle.global_best()
                gmin_cache[ds_label] = float(oracle._y_raw.min())
            gb, gmin = gb_cache[ds_label], gmin_cache[ds_label]

            with open(per_seed_path) as f:
                seed_metrics = json.load(f)
            for i, m in enumerate(seed_metrics):
                row = {"dataset": ds_label, "condition": condition, "seed": i,
                       "global_best": gb, "global_min": gmin}
                row.update(m)
                rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(COMBINED / "metrics_summary.csv", index=False)
print(f"Wrote {len(df)} rows to {COMBINED/'metrics_summary.csv'}")
print(df.groupby(["dataset", "condition"]).size().to_string())
