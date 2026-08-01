"""
mab_noise_diagnostic.py — repeat-seed noise diagnostic for the mAb/excipient
oracle, run BEFORE any further mAb evolution/generalization work (per the
project's agreed sequencing): determine whether the mAb noise floor
(SE(mean_margin)~0.057 at n=75 training campaigns vs. observed top-AF
margin gaps of ~0.011) is PIPELINE noise (GP-fit / UNSGA3-candidate-
generation stochasticity) or DOMAIN noise (intrinsic to the oracle/
campaign), since these have different fixes: pipeline noise reproduces on
every dataset regardless of how "real" or "sparse" it is, so generalization
testing against it would just reproduce the same unresolved problem N
times without telling you anything new; domain noise is a property of the
mAb oracle itself and means the fix has to be a noise-robust fitness metric
or a much larger n_campaigns, not a pipeline change.

Method: fix ONE AF (gen6_child0, the coatings evolution winner — chosen
because its formula is simple/fixed and its own scoring logic has zero
internal randomness, so any HV variance we see is attributable entirely to
the upstream candidate-generation pipeline, not the AF itself) and ONE
mAb campaign (fixed X_init/Y_init from a real training log — same initial
points every run). Vary:
  - GP-fit seed only (torch.manual_seed, controls SingleTaskGP
    initialization/fit_gpytorch_mll restarts and SobolQMCNormalSampler
    draws), holding the UNSGA3 seed fixed.
  - UNSGA3 seed only (the numpy Generator run_mo_campaign derives from its
    own `seed` argument, controls reference-direction sampling and
    evolutionary population init/selection), holding the GP-fit seed
    fixed.
These are decoupled without modifying full_replay.py/excipient_campaign_mo.py
at all: torch.manual_seed() sets torch's GLOBAL RNG state (used by GP
fitting), while run_mo_campaign's `seed` argument only drives its own
LOCAL `np.random.default_rng(seed)` instance (used for UNSGA3) — calling
torch.manual_seed(gp_seed) once, then run_mo_campaign(..., seed=unsga_seed)
directly (bypassing run_2b_campaign's single-seed wrapper, which ties both
to the same integer), gives independent control over each axis.

Interpretation: compare the seed-only-driven final_hv spread (same
campaign, same AF, different seeds) against the cross-campaign final_hv
spread already measured for baseline_hvs in the mAb evolution run (CV~50%,
i.e. std~3977 on mean~8010, see evolve_af_v2.py's GAMMA_DEFAULTS comment
and this project's earlier CV verification). If seed-only spread is a
small fraction of that cross-campaign spread, domain noise dominates and
pipeline seeding is not the problem. If seed-only spread is comparable in
size, pipeline noise is a real, fixable contributor and should be
addressed (e.g. more fit_gpytorch_mll restarts, fixed seeding across the
evaluate_af_2b's campaign loop) before trusting any mAb fitness number.

Usage:
    python mab_noise_diagnostic.py --n_seeds 15
    python mab_noise_diagnostic.py --n_seeds 15 --af_id trust_only \\
        --population_path evolution_runs/run_v2_coatings_gamma001_v2/final_population.json
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
import torch

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

from excipient_campaign_mo import run_mo_campaign
from full_replay import reconstruct_oracle, strategy_evolved_af
from strategy_ls_na_egbo import strategy_mo_egbo_novelty

HERE = pathlib.Path(__file__).parent


def run_one_af(disc_oracle, X_init, Y_init, budget, batch_size, af_code,
               gp_seed: int, unsga_seed: int) -> float:
    """One campaign under the evolved AF, decoupled GP-fit seed vs. UNSGA3
    seed — see module docstring for why this decoupling works without
    touching full_replay.py."""
    torch.manual_seed(gp_seed)
    strategy_kwargs = {"af_code": af_code, "budget": budget,
                        "sandbox_log_dir": None, "hv_history": []}
    result = run_mo_campaign(disc_oracle, X_init, Y_init, budget,
                              strategy_evolved_af, strategy_kwargs,
                              batch_size=batch_size, seed=unsga_seed)
    return result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")


def run_one_baseline(disc_oracle, X_init, Y_init, budget, batch_size,
                      gp_seed: int, unsga_seed: int) -> float:
    """Same campaign/seed pair, but strategy_mo_egbo_novelty (the
    production baseline) instead of the evolved AF — needed to compute the
    PAIRED relative margin at each seed, not just the evolved AF's raw HV.
    Baseline and evolved AF share the same candidate-generation code path
    (GP fit + qLogNEHVI + UNSGA3) up to final selection, so if that shared
    path consumes randomness identically, a shared seed could make much of
    the pipeline noise measured in Axis 1/2 CANCEL in the paired
    difference even though it doesn't cancel in either raw HV alone — this
    function exists to check that directly rather than assume it."""
    torch.manual_seed(gp_seed)
    result = run_mo_campaign(disc_oracle, X_init, Y_init, budget,
                              strategy_mo_egbo_novelty, {},
                              batch_size=batch_size, seed=unsga_seed)
    return result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population_path",
                     default=str(HERE / "evolution_runs" / "run_v2_coatings_gamma001_v2" /
                                 "final_population.json"))
    ap.add_argument("--af_id", default="gen6_child0",
                     help="Which AF from --population_path to fix for this "
                          "diagnostic (default: the coatings evolution winner, "
                          "chosen because its own scoring logic has zero "
                          "internal randomness).")
    ap.add_argument("--train_dir", default=str(HERE / "training_logs" / "train"),
                     help="mAb/excipient training logs dir — one campaign is "
                          "picked from here and held fixed for the whole diagnostic.")
    ap.add_argument("--campaign_index", type=int, default=0,
                     help="Which campaign file (sorted order) to fix.")
    ap.add_argument("--n_seeds", type=int, default=15,
                     help="Number of seed draws per axis (GP-fit, UNSGA3).")
    ap.add_argument("--fixed_gp_seed", type=int, default=0)
    ap.add_argument("--fixed_unsga_seed", type=int, default=0)
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    out_path = args.out_path or str(HERE / "mab_noise_diagnostic_results.json")

    population = json.load(open(args.population_path))
    af = next((p for p in population if p["id"] == args.af_id), None)
    if af is None:
        raise SystemExit(f"AF id {args.af_id!r} not found in {args.population_path} "
                          f"(available: {[p['id'] for p in population]})")
    af_code = af["code"]
    print(f"Fixed AF: {args.af_id} (LOC={af['loc']})")

    files = sorted(pathlib.Path(args.train_dir).glob("*.json"))
    log = json.load(open(files[args.campaign_index]))
    print(f"Fixed campaign: {files[args.campaign_index].name}")

    disc_oracle = reconstruct_oracle(log, oracle_family="excipient")
    X_init = np.array(log["X_init"])
    Y_init = np.array(log["Y_init"])
    budget = log["budget"]
    batch_size = log["batch_size"]

    print(f"\n--- Axis 1: vary GP-fit seed (0..{args.n_seeds - 1}), "
          f"UNSGA3 seed fixed at {args.fixed_unsga_seed} (evolved AF only) ---")
    gp_axis_hvs = []
    for gp_seed in range(args.n_seeds):
        hv = run_one_af(disc_oracle, X_init, Y_init, budget, batch_size, af_code,
                         gp_seed=gp_seed, unsga_seed=args.fixed_unsga_seed)
        gp_axis_hvs.append(hv)
        print(f"  gp_seed={gp_seed:>3}: final_hv={hv:.2f}")

    print(f"\n--- Axis 2: vary UNSGA3 seed (0..{args.n_seeds - 1}), "
          f"GP-fit seed fixed at {args.fixed_gp_seed} (evolved AF only) ---")
    unsga_axis_hvs = []
    for unsga_seed in range(args.n_seeds):
        hv = run_one_af(disc_oracle, X_init, Y_init, budget, batch_size, af_code,
                         gp_seed=args.fixed_gp_seed, unsga_seed=unsga_seed)
        unsga_axis_hvs.append(hv)
        print(f"  unsga_seed={unsga_seed:>3}: final_hv={hv:.2f}")

    print(f"\n--- Axis 3: JOINT seed (gp_seed=unsga_seed=s, matching "
          f"production's single-seed-per-campaign convention), evolved AF "
          f"AND baseline both run at each s, to check whether pipeline "
          f"noise CANCELS in the paired relative margin ---")
    joint_af_hvs, joint_baseline_hvs, joint_margins = [], [], []
    for s in range(args.n_seeds):
        hv_af = run_one_af(disc_oracle, X_init, Y_init, budget, batch_size, af_code,
                            gp_seed=s, unsga_seed=s)
        hv_base = run_one_baseline(disc_oracle, X_init, Y_init, budget, batch_size,
                                    gp_seed=s, unsga_seed=s)
        margin = (hv_af - hv_base) / hv_base if hv_base else float("nan")
        joint_af_hvs.append(hv_af)
        joint_baseline_hvs.append(hv_base)
        joint_margins.append(margin)
        print(f"  seed={s:>3}: af_hv={hv_af:.2f}  baseline_hv={hv_base:.2f}  "
              f"margin={margin:+.4f}")

    gp_arr = np.array(gp_axis_hvs)
    unsga_arr = np.array(unsga_axis_hvs)
    joint_af_arr = np.array(joint_af_hvs)
    joint_base_arr = np.array(joint_baseline_hvs)
    margin_arr = np.array(joint_margins)

    print(f"\n{'='*70}")
    print(f"GP-fit-seed axis (AF only):    mean={gp_arr.mean():.2f}  std={gp_arr.std():.2f}  "
          f"CV={100*gp_arr.std()/gp_arr.mean():.2f}%")
    print(f"UNSGA3-seed axis (AF only):    mean={unsga_arr.mean():.2f}  std={unsga_arr.std():.2f}  "
          f"CV={100*unsga_arr.std()/unsga_arr.mean():.2f}%")
    print(f"\nJoint-seed AF hv:              mean={joint_af_arr.mean():.2f}  "
          f"std={joint_af_arr.std():.2f}  CV={100*joint_af_arr.std()/joint_af_arr.mean():.2f}%")
    print(f"Joint-seed baseline hv:        mean={joint_base_arr.mean():.2f}  "
          f"std={joint_base_arr.std():.2f}  CV={100*joint_base_arr.std()/joint_base_arr.mean():.2f}%")
    print(f"Joint-seed PAIRED margin:      mean={margin_arr.mean():+.4f}  "
          f"std={margin_arr.std():.4f}  (this SD is the seed-only contribution "
          f"to the SE(mean_margin) figure used in evaluate_af_2b's fitness — "
          f"compare directly against the ~0.057 SE(mean_margin) already "
          f"measured across 75 DIFFERENT campaigns, and against the ~0.011 "
          f"observed margin gap between top AFs)")
    print(f"\nIf margin std above is small relative to 0.057, pairing cancels "
          f"most of the pipeline noise and the mAb SE is dominated by genuine "
          f"cross-campaign domain differences, not seed noise — domain-noise "
          f"fix (median/bootstrap fitness, more campaigns) is the right next "
          f"step. If it's comparable to or larger than 0.011, pipeline noise "
          f"alone can already swamp the AF-vs-AF signal even before domain "
          f"noise is considered, and averaging fitness over a few fixed "
          f"seeds per campaign should be added regardless of the domain-"
          f"noise finding.")

    with open(out_path, "w") as f:
        json.dump({
            "af_id": args.af_id, "campaign_file": files[args.campaign_index].name,
            "n_seeds": args.n_seeds, "fixed_gp_seed": args.fixed_gp_seed,
            "fixed_unsga_seed": args.fixed_unsga_seed,
            "gp_axis_hvs": gp_axis_hvs, "unsga_axis_hvs": unsga_axis_hvs,
            "joint_af_hvs": joint_af_hvs, "joint_baseline_hvs": joint_baseline_hvs,
            "joint_margins": joint_margins,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
