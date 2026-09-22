"""
select_full_pool_champion.py — required post-processing step for any v7
run, not an optional extra: v7's per-generation resampling means
population[0] at the final generation is whoever won THAT generation's
random K-campaign draw, not necessarily the genuinely best individual
over the whole run (see evolve_af_v7.py's module docstring and v7/
README.md's Results section — a real run showed fitness swinging from
~-1M to +51 generation to generation purely from resampling noise).

This re-scores every member of <run_dir>/final_population.json on the
FULL per-domain campaign pool (all --n_campaigns_* campaigns, not a K
subsample) and writes the single best (by full-pool fitness) to
<run_dir>/best_af_fullpool.py — THIS is the file that should go on to
held-out validation (validate_heldout.py), not the run's own
best_af.py (which is just whichever candidate happened to be
population[0] when the run stopped, i.e. the last generation's lucky
subsample winner).

Usage:
    python select_full_pool_champion.py --run_dir evolution_runs/run1
"""

import argparse
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
    _LLM_AF_EVO / "v6" / "src",
    _LLM_AF_EVO / "v7" / "src",
):
    sys.path.insert(0, str(_p))

import json

from evolve_af_v7 import evaluate_af_multi_domain, compute_baseline_hvs, \
    load_training_campaigns, GAMMA_V6_DEFAULT

_V6_EXPERIMENTS = _LLM_AF_EVO / "v6" / "experiments"

# Same 3 training domains/noise levels as v6/v7's gate — see v6/README.md.
DOMAIN_SPECS = [
    ("zdt1", _V6_EXPERIMENTS / "training_logs_zdt1" / "train", "zdt1"),
    ("dtlz2_3obj", _V6_EXPERIMENTS / "training_logs_dtlz2_3obj" / "train", "dtlz2"),
    ("zdt3", _V6_EXPERIMENTS / "training_logs_zdt3" / "train", "zdt3"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_campaigns", type=int, default=16,
                     help="Full-pool size per domain — should match the run's own "
                          "--n_campaigns_zdt1/etc. (default 16, i.e. 'load every "
                          "available training campaign' — see load_training_campaigns).")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    pop = json.loads((run_dir / "final_population.json").read_text())
    print(f"{len(pop)} final population members; re-scoring each on the FULL "
          f"campaign pool per domain (not the per-generation K-subsample) to pick "
          f"the genuinely best one.\n")

    rng = np.random.default_rng(0)
    domain_configs = {}
    for name, train_dir, family in DOMAIN_SPECS:
        logs = load_training_campaigns(train_dir, args.n_campaigns, rng)
        bhv = compute_baseline_hvs(logs, oracle_family=family)
        domain_configs[name] = {"training_logs": logs, "baseline_hvs": bhv,
                                 "oracle_family": family, "n_fitness_seeds": 1}

    results = []
    for p in pop:
        r = evaluate_af_multi_domain(p["code"], domain_configs, GAMMA_V6_DEFAULT)
        results.append({"id": p["id"], **r})
        print(f"{p['id']:20s} full-pool fitness={r['fitness']:+.4f} avg_z={r['mean_margin']:+.4f} "
              f"domains_passed={r['win_rate']:.2f} domain_zs={r['domain_zs']}")

    best = max(results, key=lambda r: r["fitness"])
    best_code = next(p["code"] for p in pop if p["id"] == best["id"])
    print(f"\nTrue full-pool champion: {best['id']} fitness={best['fitness']:+.4f}")
    out_path = run_dir / "best_af_fullpool.py"
    out_path.write_text(best_code)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
