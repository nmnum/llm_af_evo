"""
track_hv_trajectory.py — logs the FULL per-batch hv_trajectory (not just
final HV) for gen6_child0_tuned vs hint_fixed_ucb, to answer: does the
normalised AF reach a given HV level earlier (faster convergence), or does
it plateau at the same rate and only pull ahead right at the end?

Uses run_mo_campaign directly — it already returns hv_trajectory per batch,
no reimplementation needed (unlike track_front_range.py, which needed a
custom loop because front_range isn't part of run_mo_campaign's return
value).

Usage:
    python track_hv_trajectory.py --n_campaigns 20 --budget 20
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import pathlib
import sys

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
from excipient_campaign_mo import run_mo_campaign, make_shared_inits

import torch
from tunable_synthetic_oracle import TunableSyntheticMOOracle
from full_replay import strategy_evolved_af

HERE = pathlib.Path(__file__).parent

SEED_HINT_FIXED_UCB = '''
def score_pool(context):
    """Fixed-weight UCB, RAW (non-front-range-normalised) sigma."""
    names = context["objective_names"]
    beta = 2.0
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        scores.append(mu_sum + beta * sigma_sum)
    return scores
'''.strip("\n")

_GEN6_CHILD0_TUNED = '''
def score_pool(context):
    """Front-range-normalised sigma UCB (gen6_child0). beta=15.0."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    beta = 15.0
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        scores.append(mu_sum + beta * sigma_norm)
    return scores
'''.strip("\n")

CONDITIONS = {
    "hint_fixed_ucb": SEED_HINT_FIXED_UCB,
    "gen6_child0_tuned": _GEN6_CHILD0_TUNED,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_campaigns", type=int, default=20)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plateau_sharpness", type=float, default=5.0)
    ap.add_argument("--noise_level", type=float, default=0.08)
    ap.add_argument("--noise_mode", default="proportional")
    ap.add_argument("--scale2", type=float, default=3.0)
    ap.add_argument("--out_path", default=str(HERE / "hv_trajectory_trace.json"))
    args = ap.parse_args()

    oracle = TunableSyntheticMOOracle.build(
        plateau_sharpness=args.plateau_sharpness, noise_level=args.noise_level,
        noise_mode=args.noise_mode, scale2=args.scale2, seed=args.seed)
    inits = make_shared_inits(oracle, args.n_campaigns, args.n_init, rng_seed=args.seed)

    all_trajs = {cond: [] for cond in CONDITIONS}
    for cond_name, af_code in CONDITIONS.items():
        print(f"\n{cond_name}...")
        for i, (X_init, Y_init) in enumerate(inits):
            torch.manual_seed(i)
            result = run_mo_campaign(oracle, X_init, Y_init, args.budget,
                                      strategy_evolved_af,
                                      {"af_code": af_code, "budget": args.budget},
                                      batch_size=args.batch_size, seed=i)
            all_trajs[cond_name].append(result["hv_trajectory"])

    n_batches = len(all_trajs["hint_fixed_ucb"][0])
    print(f"\n{'='*70}\nMean HV by batch index (averaged over {args.n_campaigns} campaigns):")
    print(f"{'batch':>6} {'hint_fixed_ucb':>16} {'gen6_child0_tuned':>18} {'lead':>10}")
    crossed_at = None
    for b in range(n_batches):
        h = np.mean([t[b] for t in all_trajs["hint_fixed_ucb"]])
        g = np.mean([t[b] for t in all_trajs["gen6_child0_tuned"]])
        lead = g - h
        marker = ""
        if lead > 0 and crossed_at is None:
            crossed_at = b
            marker = "  <- first batch tuned leads"
        print(f"{b:>6} {h:>16.4f} {g:>18.4f} {lead:>+10.4f}{marker}")

    with open(args.out_path, "w") as f:
        json.dump({"n_campaigns": args.n_campaigns, "budget": args.budget,
                    "trajectories": all_trajs}, f, indent=2)
    print(f"\nSaved trajectories to {args.out_path}")


if __name__ == "__main__":
    main()
