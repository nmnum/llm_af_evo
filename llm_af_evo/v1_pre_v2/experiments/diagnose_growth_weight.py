"""
diagnose_growth_weight.py — instruments modification (1)'s growth_weight
per batch to check the hypothesis raised after run_growth_aware_v2.py's
full-scale result (growth_aware_v2 fires but doesn't help, effect ~
unchanged from the dead-mechanism baseline): that boundary_std collapses
toward 0 (growth_weight -> 1) within the first few batches, because an
active-learning GP's posterior uncertainty is lowest exactly at the point
that's already sitting on the front's edge (where the campaign has been
sampling most), for the same underlying reason front_range grows
monotonically — so the "extra" exploration boost modification (1) adds is
real but too small and too fast-decaying to counteract gen6_child0's
persistent under-exploration in the back half of the campaign.

Cheap by design: 2 domain seeds x 3 campaigns, growth_aware_v2 condition
only (no MixedLM stage — this is a mechanism check, not a confirmatory
statistical test) — just needs enough batches per campaign to see whether
growth_weight decays and where it stabilises.
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
from run_growth_aware_v2 import _GEN6_CHILD0_GROWTH_AWARE_V2

import full_replay

DOMAIN_SEEDS = [42, 43]
N_CAMPAIGNS = 3


def main():
    full_replay.enable_boundary_debug_log()
    af_code = _GEN6_CHILD0_GROWTH_AWARE_V2.format(beta=2)

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

    rows = full_replay.get_boundary_debug_log()
    df = pd.DataFrame(rows)
    df["domain_seed"] = df["tag"].apply(lambda t: t[0])
    df["campaign"] = df["tag"].apply(lambda t: t[1])
    df = df.drop(columns=["tag"])
    out_path = HERE / "growth_weight_debug.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # batch index from n_obs: n_init=10, batch_size=5 -> batch = (n_obs-10)/5
    df["batch"] = ((df["n_obs"] - 10) / 5).round().astype(int)

    print(f"\n{'='*70}\ngrowth_weight by batch (mean over domain_seeds x campaigns x "
          f"objectives)\n{'='*70}")
    summary = (df.groupby("batch")
                 .agg(growth_weight_mean=("growth_weight", "mean"),
                      growth_weight_max=("growth_weight", "max"),
                      boundary_std_mean=("boundary_std", "mean"),
                      front_range_mean=("front_range", "mean"))
                 .reset_index())
    print(summary.to_string(index=False))

    print(f"\n{'='*70}\ngrowth_weight by objective (pooled over all batches)\n{'='*70}")
    by_obj = (df.groupby("objective")
                .agg(growth_weight_mean=("growth_weight", "mean"),
                     growth_weight_max=("growth_weight", "max"))
                .reset_index())
    print(by_obj.to_string(index=False))


if __name__ == "__main__":
    main()
