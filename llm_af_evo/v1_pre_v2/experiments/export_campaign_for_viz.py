"""
export_campaign_for_viz.py — run a real AF against a real (or synthetic)
oracle through the actual BoTorch-driven campaign loop (full_replay.py's
strategy_evolved_af, same one track_front_range.py / run_tunable_domain_
generalization.py use), and dump the FULL per-batch objective-space
trajectory (not just front_range) as JSON that pareto_front_explorer.html
can load directly.

This is the "stop hardcoding synthetic samples, show me what an actual AF
run produced" counterpart to pareto_front_explorer.html's built-in DTLZ2
generator — same JSON consumer, real data producer.

Oracles supported (--oracle):
    tunable   TunableSyntheticMOOracle  (2 obj, tunable scale/noise/plateau)
    zdt1      DiscreteSyntheticMOOracle.build_zdt1  (2 obj, fixed)
    dtlz2     DiscreteSyntheticMOOracle.build_dtlz2 (--n_obj objectives, 2-4)
    coatings  DiscreteADACoatingsOracle  (2 obj, real ADA coatings data —
              needs coatings CSVs in llm_af_evo/shared/coatings_data/,
              downloaded separately from github.com/berlinguette/ada; not
              bundled in this repo, so this oracle will FileNotFoundError
              on a fresh checkout until you add them)
    mab       DiscreteMOExcipientOracle  (3 obj: Tm/kD/viscosity, real mAb data)

AFs supported (--af): trust_only, novelty_only, phase_decaying_ucb,
ehvi_approx, mc_hvi_approx (all af_interface.SEED_PROGRAMS keys written
generically over context["objective_names"], so they run on any oracle
above), plus the two ad-hoc AFs from track_front_range.py (hint_fixed_ucb,
gen6_child0_tuned, also objective-name-agnostic). fixed_ucb and
ucb_plus_novelty are ALSO SEED_PROGRAMS keys but hardcode the mAb objective
names (Tm/kD/viscosity) — they only run with --oracle mab; resolve_af_code()
raises a clear error rather than letting them KeyError deep in the sandbox
on any other oracle. Or point --af_file at a .py file defining its own
score_pool(context) to run anything else.

Requires torch/botorch/gpytorch (see repo requirements.txt) — this runs the
real harness, not a lightweight stand-in.

Usage:
    python export_campaign_for_viz.py --oracle dtlz2 --n_obj 3 \
        --af gen6_child0_tuned --out campaign_dtlz2_3obj.json
    python export_campaign_for_viz.py --oracle mab --af fixed_ucb \
        --out campaign_mab.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_V1 = _HERE.parent
_LLM_AF_EVO = _V1.parent
_ROOT = _LLM_AF_EVO.parent
for _p in (_ROOT, _LLM_AF_EVO / "shared", _V1 / "src", _V1 / "experiments"):
    sys.path.insert(0, str(_p))

import torch  # noqa: E402
from excipient_campaign_mo import pareto_front_of, snap_query_mo  # noqa: E402
from full_replay import strategy_evolved_af  # noqa: E402
from af_interface import SEED_PROGRAMS  # noqa: E402

# Same two ad-hoc AF strings track_front_range.py / run_tunable_domain_
# generalization.py use — duplicated here for the same reason those files
# duplicate them (throwaway-diagnostic scripts, not a shared module).
_HINT_FIXED_UCB = '''
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

_EXTRA_AFS = {"hint_fixed_ucb": _HINT_FIXED_UCB, "gen6_child0_tuned": _GEN6_CHILD0_TUNED}


def build_oracle(name: str, n_obj: int, seed: int,
                  plateau_sharpness: float, noise_level: float,
                  noise_mode: str, scale2: float):
    if name == "tunable":
        from tunable_synthetic_oracle import TunableSyntheticMOOracle
        return TunableSyntheticMOOracle.build(
            plateau_sharpness=plateau_sharpness, noise_level=noise_level,
            noise_mode=noise_mode, scale2=scale2, seed=seed)
    if name == "zdt1":
        from synthetic_mo_oracle import DiscreteSyntheticMOOracle
        return DiscreteSyntheticMOOracle.build_zdt1(seed=seed)
    if name == "dtlz2":
        from synthetic_mo_oracle import DiscreteSyntheticMOOracle
        return DiscreteSyntheticMOOracle.build_dtlz2(n_obj=n_obj, seed=seed)
    if name == "coatings":
        from ada_coatings_oracle import DiscreteADACoatingsOracle
        return DiscreteADACoatingsOracle.build()
    if name == "mab":
        # DiscreteMOExcipientOracle.build() takes a MultiObjectiveExcipientOracle
        # instance, not a bare seed — same two-step construction every other
        # caller in this repo uses (run_mock_subset.py, debug_generation.py,
        # run_unsga3_pool_pilot.py, ...): build the continuous oracle first,
        # then discretise it.
        from excipient_oracle_mo import MultiObjectiveExcipientOracle
        oracle_full = MultiObjectiveExcipientOracle(
            protein="mAb_aggregation", tm_noise=0.013, kd_noise=0.096,
            viscosity_noise=0.10, seed=seed)
        return oracle_full.make_discrete_oracle(n_samples=500, seed=seed)
    raise ValueError(f"unknown --oracle {name!r}")


# af_interface.SEED_FIXED_UCB and SEED_UCB_PLUS_NOVELTY hardcode gp["Tm"],
# gp["kD"], gp["viscosity"] rather than iterating context["objective_names"]
# (unlike trust_only/novelty_only/phase_decaying_ucb/ehvi_approx/
# mc_hvi_approx, which are all objective-name-agnostic) — they were written
# for the mAb domain specifically and KeyError on anything else's objective
# names. Caught here with a clear message instead of letting it surface as
# a cryptic "AF program exited nonzero: KeyError: 'Tm'" from deep inside
# the sandbox subprocess.
_MAB_ONLY_AFS = {"fixed_ucb", "ucb_plus_novelty"}


def resolve_af_code(af: str, af_file: str, oracle_name: str) -> str:
    if af_file:
        return pathlib.Path(af_file).read_text()
    if af in _MAB_ONLY_AFS and oracle_name != "mab":
        raise ValueError(
            f"--af {af!r} is hardcoded to the mAb objective names (Tm/kD/viscosity) "
            f"and only runs with --oracle mab, not --oracle {oracle_name!r}. Use "
            "trust_only, novelty_only, phase_decaying_ucb, ehvi_approx, "
            "mc_hvi_approx, hint_fixed_ucb, or gen6_child0_tuned instead — those are "
            "all written generically over context['objective_names'].")
    if af in SEED_PROGRAMS:
        return SEED_PROGRAMS[af]
    if af in _EXTRA_AFS:
        return _EXTRA_AFS[af]
    raise ValueError(
        f"unknown --af {af!r}; choose from {sorted(SEED_PROGRAMS) + sorted(_EXTRA_AFS)} "
        "or pass --af_file pointing at a script defining score_pool(context)")


def run_campaign(oracle, budget, n_init, batch_size, seed, af_code):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    bounds = oracle.bounds()
    directions = oracle.objective_directions()
    names = oracle.objective_names()

    init_idx = rng.choice(len(oracle), size=n_init, replace=False)
    X_obs = oracle._X_raw[init_idx].copy()
    Y_obs = oracle._Y_raw[init_idx].copy() if hasattr(oracle, "_Y_raw") else np.array(
        [oracle.query_mo(oracle._X_raw[i])[0] for i in init_idx])
    oracle._queried = set(init_idx.tolist())

    n_batches = max(1, (budget - n_init) // batch_size)

    def snapshot(batch_idx):
        pf_idx = set(pareto_front_of(Y_obs, directions=directions).tolist())
        return {
            "batch": batch_idx,
            "n_obs": len(Y_obs),
            "Y": Y_obs.tolist(),
            "pareto_mask": [i in pf_idx for i in range(len(Y_obs))],
        }

    batches = [snapshot(0)]
    for b in range(n_batches):
        candidates, _ = strategy_evolved_af(
            oracle=oracle, X_obs=X_obs, Y_obs=Y_obs, bounds=bounds,
            batch_size=batch_size, rng=rng, af_code=af_code, budget=budget)
        new_x, new_y = snap_query_mo(candidates[:batch_size], oracle)
        if new_x:
            X_obs = np.vstack([X_obs, new_x])
            Y_obs = np.vstack([Y_obs, new_y])
        batches.append(snapshot(b + 1))

    return names, directions, batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="dtlz2",
                    choices=["tunable", "zdt1", "dtlz2", "coatings", "mab"])
    ap.add_argument("--n_obj", type=int, default=3, help="dtlz2 only, 2-4")
    ap.add_argument("--af", default="fixed_ucb")
    ap.add_argument("--af_file", default=None,
                     help="path to a .py file defining score_pool(context); overrides --af")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plateau_sharpness", type=float, default=5.0, help="tunable oracle only")
    ap.add_argument("--noise_level", type=float, default=0.08, help="tunable oracle only")
    ap.add_argument("--noise_mode", default="proportional", help="tunable oracle only")
    ap.add_argument("--scale2", type=float, default=3.0, help="tunable oracle only")
    ap.add_argument("--out", default=str(_HERE / "campaign_export.json"))
    args = ap.parse_args()

    oracle = build_oracle(args.oracle, args.n_obj, args.seed,
                           args.plateau_sharpness, args.noise_level,
                           args.noise_mode, args.scale2)
    af_code = resolve_af_code(args.af, args.af_file, args.oracle)
    af_label = args.af_file or args.af

    names, directions, batches = run_campaign(
        oracle, args.budget, args.n_init, args.batch_size, args.seed, af_code)

    out = {
        "meta": {
            "oracle": args.oracle, "af": af_label, "budget": args.budget,
            "n_init": args.n_init, "batch_size": args.batch_size, "seed": args.seed,
            "objective_names": names, "objective_directions": directions,
        },
        "batches": batches,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    n_front_final = sum(batches[-1]["pareto_mask"])
    print(f"{args.oracle}/{af_label}: {len(names)} objectives, "
          f"{len(batches)-1} batches, final n_obs={batches[-1]['n_obs']}, "
          f"final front size={n_front_final}")
    print(f"Wrote {args.out} — load it in pareto_front_explorer.html")


if __name__ == "__main__":
    main()
