"""
run_2b_diagnostic.py — the two cheap tests before committing to a full 2b
evolution run:

Step A: time a real 2b campaign (evolved AF as the acquisition function,
        via full_replay.strategy_evolved_af) on a handful of held-out
        campaigns. This gives the REAL per-batch/per-campaign cost with
        the evolved AF doing the scoring, not an estimate from timing the
        qLogNEHVI baseline (generate_training_set.py's pilot measured a
        different thing: the baseline's OWN acquisition-optimisation cost,
        which is dominated by optimize_acqf's internal repeated
        evaluations for candidate GENERATION — that part is NOT removed by
        swapping in the evolved AF, since the same candidate generation is
        still needed to produce a pool for the evolved AF to score).

Baseline: EGBO-novelty (unmodified) run through the same 2b loop, same
        campaigns, same seeds, via full_replay.run_baseline_campaign — the
        reference point every evolved AF's mean_final_hv_2b needs to be
        judged against. Without this, an evolved AF's raw HV number has no
        scale (previously the diagnostic printed final_hv per evolved AF
        with nothing to compare it to).

Step B: evaluate every AF in run1's final population under 2b on the SAME
        small set of held-out campaigns, report how many beat the
        baseline's mean final HV, and compare the 2b ranking (by mean
        final HV) to the 2a ranking (by their already-known fitness) via
        Spearman correlation. Spearman < 0.5 -> the 2a proxy is leaky, 2b
        evolution is worth the cost. Spearman > 0.8 -> 2b won't change the
        outcome; the problem is elsewhere (search space / AF grammar /
        baseline strength), not the proxy.

Usage:
    python run_2b_diagnostic.py --n_campaigns 5
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from scipy.stats import spearmanr

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
from full_replay import run_2b_campaign, run_baseline_campaign

HERE = pathlib.Path(__file__).parent


def load_heldout_logs(heldout_dir: pathlib.Path, n: int) -> list:
    files = sorted(pathlib.Path(heldout_dir).glob("*.json"))[:n]
    return [json.load(open(f)) for f in files]


def step_a_timing(af_code: str, logs: list, sandbox_log_dir=None) -> dict:
    print(f"Step A: timing {len(logs)} full 2b campaigns with the evolved AF...")
    per_campaign_times = []
    for i, log in enumerate(logs):
        t0 = time.perf_counter()
        result = run_2b_campaign(af_code, log, seed=i, sandbox_log_dir=sandbox_log_dir)
        elapsed = time.perf_counter() - t0
        per_campaign_times.append(elapsed)
        n_batches = len(result["hv_trajectory"])
        print(f"  campaign {i}: {elapsed:.2f}s total, "
              f"{elapsed / max(n_batches, 1):.2f}s/batch ({n_batches} batches), "
              f"final_hv={result['final_hv']:.1f}")

    mean_campaign = float(np.mean(per_campaign_times))
    return {"per_campaign_seconds": per_campaign_times, "mean_campaign_seconds": mean_campaign}


def compute_baseline(logs: list) -> dict:
    """
    EGBO-novelty's own 2b performance on the SAME campaigns/seeds used for
    the evolved AFs — the reference point Step B was missing. Without
    this, an evolved AF's mean_final_hv_2b is a number with no scale: you
    can't tell "2701" apart from "beats the thing we're trying to beat"
    without knowing what the baseline itself scores here.
    """
    print(f"\nBaseline: running EGBO-novelty (unmodified) under 2b on "
          f"{len(logs)} held-out campaigns for reference...")
    final_hvs = []
    for i, log in enumerate(logs):
        result = run_baseline_campaign(log, seed=i)
        final_hvs.append(result["final_hv"])
        print(f"  campaign {i}: final_hv={result['final_hv']:.1f}")
    mean_hv = float(np.mean(final_hvs))
    print(f"Baseline mean_final_hv = {mean_hv:.1f}")
    return {"mean_final_hv": mean_hv, "final_hvs": final_hvs}


def step_b_rank_comparison(population: list, logs: list, baseline: dict,
                            sandbox_log_dir=None) -> dict:
    print(f"\nStep B: evaluating {len(population)} AFs from run1 under 2b "
          f"on {len(logs)} held-out campaigns...")
    baseline_hv = baseline["mean_final_hv"]
    print(f"  {'[baseline: EGBO-novelty]':<16} mean_final_hv_2b={baseline_hv:.1f}")
    rows = []
    n_beats_baseline = 0
    for p in population:
        final_hvs = []
        for i, log in enumerate(logs):
            result = run_2b_campaign(p["code"], log, seed=i, sandbox_log_dir=sandbox_log_dir)
            final_hvs.append(result["final_hv"])
        mean_hv = float(np.mean(final_hvs))
        beats_baseline = mean_hv > baseline_hv
        n_beats_baseline += beats_baseline
        rows.append({"id": p["id"], "fitness_2a": p["fitness"], "win_rate_2a": p["win_rate"],
                     "mean_final_hv_2b": mean_hv, "final_hvs_2b": final_hvs,
                     "beats_baseline": beats_baseline})
        print(f"  {p['id']:<16} fitness_2a={p['fitness']:.4f}  mean_final_hv_2b={mean_hv:.1f}"
              f"{'  <-- BEATS BASELINE' if beats_baseline else ''}")

    print(f"\n{n_beats_baseline}/{len(population)} evolved AFs beat EGBO-novelty's "
          f"2b mean final HV ({baseline_hv:.1f}) on these {len(logs)} campaigns.")

    fitness_2a = [r["fitness_2a"] for r in rows]
    hv_2b = [r["mean_final_hv_2b"] for r in rows]
    corr, pval = spearmanr(fitness_2a, hv_2b)
    print(f"\nSpearman(2a fitness, 2b mean final HV) = {corr:.3f} (p={pval:.4f})")
    if corr < 0.5:
        print("=> Rankings substantially differ: the 2a proxy is leaky. "
              "2b evolution is worth the cost — proceed.")
    elif corr > 0.8:
        print("=> Rankings agree closely: 2b won't change which AF looks best. "
              "The proxy isn't the problem — look at the search space (AF "
              "grammar) or accept the baseline may be near-optimal here.")
    else:
        print("=> Rankings partially agree — ambiguous. Use judgement: the "
              "population sample here is small, treat this as a lean, not proof.")
    return {"rows": rows, "spearman": float(corr), "spearman_p": float(pval),
            "baseline_mean_final_hv": baseline_hv, "n_beats_baseline": n_beats_baseline}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af_path", default=str(HERE / "evolution_runs" / "run1" / "best_af.py"))
    ap.add_argument("--population_path",
                     default=str(HERE / "evolution_runs" / "run1" / "final_population.json"))
    ap.add_argument("--heldout_dir", default=str(HERE / "training_logs" / "heldout"))
    ap.add_argument("--n_campaigns", type=int, default=5)
    ap.add_argument("--skip_step_b", action="store_true",
                     help="Run only the timing measurement (Step A), skip the "
                          "8-AF ranking comparison (Step B).")
    args = ap.parse_args()

    logs = load_heldout_logs(pathlib.Path(args.heldout_dir), args.n_campaigns)
    if not logs:
        print("No held-out logs found — run generate_training_set.py first.")
        return

    af_code = pathlib.Path(args.af_path).read_text()
    sandbox_log_dir = HERE / "evolution_runs" / "run1" / "af_code_logs_2b_diagnostic"

    t0 = time.perf_counter()
    timing = step_a_timing(af_code, logs, sandbox_log_dir=sandbox_log_dir)
    print(f"\nStep A total: {time.perf_counter() - t0:.1f}s for {len(logs)} campaigns "
          f"(mean {timing['mean_campaign_seconds']:.1f}s/campaign)")

    n_children_estimate_40 = 40
    n_children_estimate_160 = 160
    for n_camp in [8, 20, 50]:
        for n_child in [n_children_estimate_40, n_children_estimate_160]:
            hours = (timing["mean_campaign_seconds"] * n_camp * n_child) / 3600
            print(f"  Projected: {n_camp} campaigns x {n_child} children "
                  f"-> {hours:.2f} hours")

    if args.skip_step_b:
        return

    baseline = compute_baseline(logs)

    population = json.load(open(args.population_path))
    t0 = time.perf_counter()
    step_b_rank_comparison(population, logs, baseline, sandbox_log_dir=sandbox_log_dir)
    print(f"\nStep B total: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
