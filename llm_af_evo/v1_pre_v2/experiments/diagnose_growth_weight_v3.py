"""
diagnose_growth_weight_v3.py — same mechanism check as
diagnose_growth_weight.py, for modification (1b)/v3's pool-top-std growth
signal: does growth_weight sit on a comparable scale to v1's (near-1,
scale-mismatched) or does sourcing std from an unobserved pool candidate
(rather than the observed front boundary point) actually produce a
meaningfully larger, more informative growth_weight?

Cheap by design: 2 domain seeds x 3 campaigns — a mechanism check, not a
confirmatory statistical test.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
for _p in [HERE.parent / "src", HERE.parent.parent / "shared",
           HERE.parent.parent.parent]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from excipient_campaign_mo import make_shared_inits
from tunable_synthetic_oracle import TunableSyntheticMOOracle

from run_confirmatory_spec import (
    PLATEAU_SHARPNESS, NOISE_LEVEL, NOISE_MODE, SCALE2,
    compute_true_hv_and_ref, run_one_campaign_tracked,
)
from run_growth_aware_v3 import _GEN6_CHILD0_GROWTH_AWARE_V3

import full_replay

DOMAIN_SEEDS = [42, 43]
N_CAMPAIGNS = 3


def main():
    full_replay.enable_pool_top_debug_log()
    af_code = _GEN6_CHILD0_GROWTH_AWARE_V3.format(beta=2)

    for domain_seed in DOMAIN_SEEDS:
        oracle = TunableSyntheticMOOracle.build(
            plateau_sharpness=PLATEAU_SHARPNESS, noise_level=NOISE_LEVEL,
            noise_mode=NOISE_MODE, scale2=SCALE2, seed=domain_seed)
        hv_true, ref_point, directions = compute_true_hv_and_ref(oracle)
        inits = make_shared_inits(oracle, N_CAMPAIGNS, 10, rng_seed=domain_seed)

        for camp_i, (X_init, Y_init) in enumerate(inits):
            full_replay.set_boundary_debug_tag((domain_seed, camp_i))
            run_one_campaign_tracked(
                oracle, X_init, Y_init, hv_true, ref_point, directions,
                seed=camp_i, af_code=af_code)
        print(f"domain_seed={domain_seed} done")

    rows = full_replay.get_pool_top_debug_log()
    df = pd.DataFrame(rows)
    df["domain_seed"] = df["tag"].apply(lambda t: t[0])
    df["campaign"] = df["tag"].apply(lambda t: t[1])
    df = df.drop(columns=["tag"])
    out_path = HERE / "growth_weight_v3_debug.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    df["batch"] = ((df["n_obs"] - 10) / 5).round().astype(int)

    print(f"\n{'='*70}\ngrowth_weight (v3, pool-top-std) by batch (mean over "
          f"domain_seeds x campaigns x objectives)\n{'='*70}")
    summary = (df.groupby("batch")
                 .agg(growth_weight_mean=("growth_weight", "mean"),
                      growth_weight_max=("growth_weight", "max"),
                      top_candidate_std_mean=("top_candidate_std", "mean"),
                      front_range_mean=("front_range", "mean"))
                 .reset_index())
    print(summary.to_string(index=False))

    print(f"\n{'='*70}\ngrowth_weight (v3) by objective (pooled over all "
          f"batches)\n{'='*70}")
    by_obj = (df.groupby("objective")
                .agg(growth_weight_mean=("growth_weight", "mean"),
                     growth_weight_max=("growth_weight", "max"))
                .reset_index())
    print(by_obj.to_string(index=False))


if __name__ == "__main__":
    main()
