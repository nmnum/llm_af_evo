"""
Decay-schedule tuning pilot (scoped 2026-08-02): can a better-calibrated,
single continuous UCB decay schedule tie or beat baseline on mAb, replicated
on genuinely held-out campaigns -- rather than a discrete portfolio/switch
among distinct AF programs (which Rule 4 already showed is a liability: the
stagnation-triggered novelty boost in gen4_child0/gen1_child1 DROPPED below
significance under seed averaging, while the simpler progress-only decay in
gen5_child0/gen9_child0 STRENGTHENED)?

Family (generalizes the two significant survivors + the flat hint_fixed_ucb
baseline in one parametrization):

    score = sum(mu) + beta0 * (1 - progress)^p * sum(sigma)

    p=0 -> fixed UCB (hint_fixed_ucb's family, beta0=2.0 was the one that
           flipped from p=0.596 to p=0.030 significant purely from seed
           averaging -- Section 13 of the comprehensive log)
    p=1 -> linear decay (gen5_child0's family, beta0=2.0)
    p=2 -> quadratic decay (gen9_child0's family, beta0=1.5)

Deliberately excludes the stagnation term -- Rule 4 already closed that
question; re-including it here would just re-spend budget re-confirming it.

Three stages, run in one script (each stage's results are saved to disk
before the next starts, so a partial run is still readable):

  Stage A (screen):  10 TRAIN campaigns x 3 seeds x 15 configs.
                      Rank by MEDIAN margin (not mean -- per the noise
                      diagnostic's own recommendation, to avoid outlier-
                      campaign inflation).
  Stage B (confirm):  top 2 configs from Stage A, same 10 TRAIN campaigns,
                      5 seeds -- checks the Stage-A ranking isn't itself a
                      seed artifact before it ever touches held-out data.
  Held-out (test):    top 1 config from Stage B, 20 HELDOUT campaigns
                      (genuinely disjoint directory, training_logs_mAb_100/
                      heldout, never touched by Stage A/B), 3 seeds,
                      Wilcoxon signed-rank vs baseline.

Kill criterion, decided in advance: the held-out result must be BOTH
directionally positive AND p<0.05 with wins >= 12/20 to count as a real
tie-or-better result. No post-hoc threshold adjustment after seeing the
held-out numbers -- if it misses, this is a closed, negative gate, on record
as such, not grounds for another round of tuning on the same split.

Usage: python pilot_decay_schedule_mab.py [--out FILE]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.stats import wilcoxon

_LLM_AF_EVO = pathlib.Path(__file__).resolve().parent
while _LLM_AF_EVO.name != "llm_af_evo":
    _LLM_AF_EVO = _LLM_AF_EVO.parent
_ROOT = _LLM_AF_EVO.parent
_LS_NA_EGBO_ROOT = pathlib.Path("/home/nehamungale/ls_na_egbo")
for _p in (
    _ROOT,
    _LS_NA_EGBO_ROOT,
    _LLM_AF_EVO / "shared",
    _LLM_AF_EVO / "v1_pre_v2" / "src",
    _LLM_AF_EVO / "v1_pre_v2" / "experiments",
    _LLM_AF_EVO / "v2" / "src",
    _LLM_AF_EVO / "v2" / "experiments",
):
    sys.path.insert(0, str(_p))

from full_replay import run_2b_campaign, run_baseline_campaign

TRAIN_DIR = _LLM_AF_EVO / "data" / "training_logs_mAb_100" / "train"
HELDOUT_DIR = _LLM_AF_EVO / "data" / "training_logs_mAb_100" / "heldout"

BETA0_GRID = [0.5, 1.0, 1.5, 2.0, 2.5]
P_GRID = [0, 1, 2]


def make_af_code(beta0: float, p: int) -> str:
    return f'''def score_pool(context):
    """Progress-decaying UCB: sum(mu) + beta0*(1-progress)^p * sum(sigma), beta0={beta0}, p={p}."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    beta0 = {beta0}
    p = {p}
    beta = beta0 * (1.0 - progress) ** p
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        scores.append(mu_sum + beta * sigma_sum)
    return scores
'''


def seed_averaged_hv(run_with_seed, campaign_idx, n_seeds, seed_offset=0):
    hvs = []
    for s in range(n_seeds):
        r = run_with_seed(campaign_idx * 1000 + seed_offset + s)
        hvs.append(r["final_hv"])
    return float(np.mean(hvs)), hvs


def eval_baseline(logs, n_seeds, seed_offset=0):
    baseline_hvs = []
    for i, log in enumerate(logs):
        avg_hv, _ = seed_averaged_hv(
            lambda seed, log=log: run_baseline_campaign(log, seed=seed, oracle_family="excipient"),
            i, n_seeds, seed_offset)
        baseline_hvs.append(avg_hv)
    return baseline_hvs


def eval_config(code, logs, baseline_hvs, n_seeds, seed_offset=0):
    margins = []
    wins = 0
    for i, log in enumerate(logs):
        avg_hv, _ = seed_averaged_hv(
            lambda seed, log=log: run_2b_campaign(code, log, seed=seed, sandbox_log_dir=None,
                                                    oracle_family="excipient"),
            i, n_seeds, seed_offset)
        b = baseline_hvs[i]
        margin = (avg_hv - b) / abs(b)
        margins.append(margin)
        if avg_hv > b:
            wins += 1
    margins = np.array(margins)
    try:
        _, p_value = wilcoxon(margins)
    except ValueError:
        p_value = float("nan")
    return {
        "mean_margin": float(margins.mean()),
        "median_margin": float(np.median(margins)),
        "wins": wins, "n": len(logs), "p_value": float(p_value),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent /
                                          "pilot_decay_schedule_mab_results.json"))
    args = ap.parse_args()
    out_path = pathlib.Path(args.out)

    results = {"grid": [{"beta0": b, "p": p} for b in BETA0_GRID for p in P_GRID],
               "stage_a": {}, "stage_b": {}, "held_out": {}, "verdict": None}

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    # ---- Stage A: screen, 10 train campaigns x 3 seeds x 15 configs ----
    train_files = sorted(TRAIN_DIR.glob("*.json"))[:10]
    train_logs = [json.load(open(f)) for f in train_files]

    print(f"Stage A: baseline on {len(train_logs)} train campaigns x 3 seeds...", flush=True)
    baseline_a = eval_baseline(train_logs, n_seeds=3)
    results["stage_a"]["baseline_hvs"] = baseline_a
    save()

    stage_a_results = []
    for beta0 in BETA0_GRID:
        for p in P_GRID:
            code = make_af_code(beta0, p)
            summary = eval_config(code, train_logs, baseline_a, n_seeds=3)
            stage_a_results.append({"beta0": beta0, "p": p, **summary})
            print(f"  Stage A beta0={beta0} p={p}: median_margin={summary['median_margin']*100:+.2f}% "
                  f"wins={summary['wins']}/{summary['n']} p_value={summary['p_value']:.4f}", flush=True)
            results["stage_a"]["configs"] = stage_a_results
            save()

    stage_a_ranked = sorted(stage_a_results, key=lambda r: r["median_margin"], reverse=True)
    top2 = stage_a_ranked[:2]
    print(f"\nStage A top 2 by median margin: {[(c['beta0'], c['p']) for c in top2]}\n", flush=True)
    results["stage_a"]["top2"] = top2
    save()

    # ---- Stage B: confirm, same 10 train campaigns x 5 seeds, top 2 configs ----
    print("Stage B: baseline on same 10 train campaigns x 5 seeds...", flush=True)
    baseline_b = eval_baseline(train_logs, n_seeds=5)
    results["stage_b"]["baseline_hvs"] = baseline_b
    save()

    stage_b_results = []
    for cfg in top2:
        code = make_af_code(cfg["beta0"], cfg["p"])
        summary = eval_config(code, train_logs, baseline_b, n_seeds=5)
        stage_b_results.append({"beta0": cfg["beta0"], "p": cfg["p"], **summary})
        print(f"  Stage B beta0={cfg['beta0']} p={cfg['p']}: median_margin={summary['median_margin']*100:+.2f}% "
              f"wins={summary['wins']}/{summary['n']} p_value={summary['p_value']:.4f}", flush=True)
        results["stage_b"]["configs"] = stage_b_results
        save()

    stage_b_ranked = sorted(stage_b_results, key=lambda r: r["median_margin"], reverse=True)
    winner = stage_b_ranked[0]
    print(f"\nStage B winner: beta0={winner['beta0']} p={winner['p']}\n", flush=True)
    results["stage_b"]["winner"] = winner
    save()

    # ---- Held-out: 20 genuinely disjoint campaigns x 3 seeds, winner only ----
    heldout_files = sorted(HELDOUT_DIR.glob("*.json"))[:20]
    heldout_logs = [json.load(open(f)) for f in heldout_files]

    print(f"Held-out: baseline on {len(heldout_logs)} heldout campaigns x 3 seeds...", flush=True)
    baseline_ho = eval_baseline(heldout_logs, n_seeds=3, seed_offset=5000)
    results["held_out"]["baseline_hvs"] = baseline_ho
    save()

    code = make_af_code(winner["beta0"], winner["p"])
    ho_summary = eval_config(code, heldout_logs, baseline_ho, n_seeds=3, seed_offset=5000)
    results["held_out"]["winner_config"] = winner
    results["held_out"]["summary"] = ho_summary
    save()

    print(f"Held-out beta0={winner['beta0']} p={winner['p']}: "
          f"mean_margin={ho_summary['mean_margin']*100:+.2f}% median={ho_summary['median_margin']*100:+.2f}% "
          f"wins={ho_summary['wins']}/{ho_summary['n']} p_value={ho_summary['p_value']:.4f}", flush=True)

    verdict_pass = (ho_summary["mean_margin"] > 0 and ho_summary["p_value"] < 0.05
                     and ho_summary["wins"] >= 12)
    results["verdict"] = "PASS (real tie-or-better)" if verdict_pass else "FAIL (closed negative, per pre-registered kill criterion)"
    save()

    print(f"\nVerdict: {results['verdict']}")
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()
