"""
run_2b_diagnostic_v3.py — oracle-generalized copy of
v1_pre_v2/experiments/run_2b_diagnostic.py: same Step A (timing) / Step B
(rank comparison) design, but threads oracle_family through to
run_2b_campaign/run_baseline_campaign (the original hardcodes the
"excipient" default and never exposes an --oracle flag), plus a Step C
this domain specifically needs: a per-campaign baseline noise/CV check
under the REAL full-2b-pipeline (GP fit + qLogNEHVI/UNSGA3 + real batch
sequencing), not just the standalone-oracle noise check the af-evolution
branch conversation ran directly against TunableSyntheticMOOracle before
any of this plumbing existed. That earlier check confirmed the NOISE
MODEL's own CV (~1.5-2.9%); this one confirms whether that CV survives
being funneled through a real sequential BO campaign (batch selection,
GP posterior, HV computation) — a different, and the actually decision-
relevant, question before trusting this domain for evolution.

Usage:
    python run_2b_diagnostic_v3.py --n_campaigns 5 --skip_step_b
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
    _LLM_AF_EVO / "v3" / "src",
    _LLM_AF_EVO / "v3" / "experiments",
):
    sys.path.insert(0, str(_p))
from full_replay import run_2b_campaign, run_baseline_campaign

HERE = pathlib.Path(__file__).parent


def load_heldout_logs(heldout_dir: pathlib.Path, n: int) -> list:
    files = sorted(pathlib.Path(heldout_dir).glob("*.json"))[:n]
    return [json.load(open(f)) for f in files]


def step_a_timing(af_code: str, logs: list, oracle_family: str, sandbox_log_dir=None) -> dict:
    print(f"Step A: timing {len(logs)} full 2b campaigns with the evolved AF "
          f"(oracle_family={oracle_family!r})...")
    per_campaign_times = []
    for i, log in enumerate(logs):
        t0 = time.perf_counter()
        result = run_2b_campaign(af_code, log, seed=i, sandbox_log_dir=sandbox_log_dir,
                                  oracle_family=oracle_family)
        elapsed = time.perf_counter() - t0
        per_campaign_times.append(elapsed)
        n_batches = len(result["hv_trajectory"])
        print(f"  campaign {i}: {elapsed:.2f}s total, "
              f"{elapsed / max(n_batches, 1):.2f}s/batch ({n_batches} batches), "
              f"final_hv={result['final_hv']:.4f}")

    mean_campaign = float(np.mean(per_campaign_times))
    return {"per_campaign_seconds": per_campaign_times, "mean_campaign_seconds": mean_campaign}


def compute_baseline(logs: list, oracle_family: str) -> dict:
    print(f"\nBaseline: running EGBO-novelty (unmodified) under 2b on "
          f"{len(logs)} held-out campaigns for reference "
          f"(oracle_family={oracle_family!r})...")
    final_hvs = []
    for i, log in enumerate(logs):
        result = run_baseline_campaign(log, seed=i, oracle_family=oracle_family)
        final_hvs.append(result["final_hv"])
        print(f"  campaign {i}: final_hv={result['final_hv']:.4f}")
    mean_hv = float(np.mean(final_hvs))
    std_hv = float(np.std(final_hvs, ddof=1)) if len(final_hvs) > 1 else 0.0
    cv = 100 * std_hv / mean_hv if mean_hv else float("nan")
    print(f"Baseline mean_final_hv = {mean_hv:.4f}, std={std_hv:.4f}, CV={cv:.1f}%")
    return {"mean_final_hv": mean_hv, "std_final_hv": std_hv, "cv_pct": cv,
            "final_hvs": final_hvs}


def step_c_seed_noise(af_code: str, log: dict, oracle_family: str, n_seeds: int) -> dict:
    """
    Step C (new in this v3 script, not in the original run_2b_diagnostic.py):
    same-campaign, repeat-seed check — mirrors mab_noise_diagnostic.py's
    Axis 3 (paired evolved-AF-vs-baseline margin std across seeds on ONE
    fixed campaign), the check that found mAb's seed noise alone (std=0.2557)
    dwarfs the ~0.011 margin gap this project targets. Running the SAME
    check here, on this domain's own real 2b pipeline, is what actually
    tells us whether the standalone-oracle CV~1.5-2.9% check survives
    contact with real batch-sequencing/GP-fit randomness, or whether this
    domain has its own hidden seed-noise problem the standalone check
    couldn't see (that check never ran a real sequential campaign).
    """
    print(f"\nStep C: seed-noise check on ONE fixed campaign, {n_seeds} repeat seeds "
          f"(mirrors mab_noise_diagnostic.py's Axis 3)...")
    af_hvs, baseline_hvs, margins = [], [], []
    for s in range(n_seeds):
        af_hv = run_2b_campaign(af_code, log, seed=s, oracle_family=oracle_family)["final_hv"]
        base_hv = run_baseline_campaign(log, seed=s, oracle_family=oracle_family)["final_hv"]
        margin = (af_hv - base_hv) / abs(base_hv) if base_hv else 0.0
        af_hvs.append(af_hv)
        baseline_hvs.append(base_hv)
        margins.append(margin)
        print(f"  seed {s}: af_hv={af_hv:.4f} baseline_hv={base_hv:.4f} margin={margin:+.4%}")
    margin_std = float(np.std(margins, ddof=1)) if len(margins) > 1 else 0.0
    print(f"\nPaired margin across {n_seeds} seeds: mean={np.mean(margins):+.4%} "
          f"std={margin_std:.4%}")
    return {"af_hvs": af_hvs, "baseline_hvs": baseline_hvs, "margins": margins,
            "margin_std": margin_std}


def step_b_rank_comparison(population: list, logs: list, baseline: dict,
                            oracle_family: str, sandbox_log_dir=None) -> dict:
    print(f"\nStep B: evaluating {len(population)} AFs under 2b on "
          f"{len(logs)} held-out campaigns...")
    baseline_hv = baseline["mean_final_hv"]
    print(f"  {'[baseline: EGBO-novelty]':<16} mean_final_hv_2b={baseline_hv:.4f}")
    rows = []
    n_beats_baseline = 0
    for p in population:
        final_hvs = []
        for i, log in enumerate(logs):
            result = run_2b_campaign(p["code"], log, seed=i, sandbox_log_dir=sandbox_log_dir,
                                      oracle_family=oracle_family)
            final_hvs.append(result["final_hv"])
        mean_hv = float(np.mean(final_hvs))
        beats_baseline = mean_hv > baseline_hv
        n_beats_baseline += beats_baseline
        rows.append({"id": p["id"], "fitness_2a": p["fitness"], "win_rate_2a": p["win_rate"],
                     "mean_final_hv_2b": mean_hv, "final_hvs_2b": final_hvs,
                     "beats_baseline": beats_baseline})
        print(f"  {p['id']:<16} fitness_2a={p['fitness']:.4f}  mean_final_hv_2b={mean_hv:.4f}"
              f"{'  <-- BEATS BASELINE' if beats_baseline else ''}")

    print(f"\n{n_beats_baseline}/{len(population)} evolved AFs beat EGBO-novelty's "
          f"2b mean final HV ({baseline_hv:.4f}) on these {len(logs)} campaigns.")

    fitness_2a = [r["fitness_2a"] for r in rows]
    hv_2b = [r["mean_final_hv_2b"] for r in rows]
    corr, pval = spearmanr(fitness_2a, hv_2b)
    print(f"\nSpearman(2a fitness, 2b mean final HV) = {corr:.3f} (p={pval:.4f})")
    return {"rows": rows, "spearman": float(corr), "spearman_p": float(pval),
            "baseline_mean_final_hv": baseline_hv, "n_beats_baseline": n_beats_baseline}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af_path", default=None,
                     help="Defaults to the tunable domain's trust_only seed AF "
                          "(af_interface_v3.SEED_PROGRAMS['trust_only']) if not given — "
                          "there's no run1/best_af.py equivalent yet for this domain.")
    ap.add_argument("--population_path", default=None)
    ap.add_argument("--heldout_dir", default=str(HERE / "training_logs_tunable" / "heldout"))
    ap.add_argument("--oracle", default="tunable")
    ap.add_argument("--n_campaigns", type=int, default=5)
    ap.add_argument("--n_seed_repeats", type=int, default=8,
                     help="Repeat-seed count for Step C's noise check.")
    ap.add_argument("--skip_step_b", action="store_true")
    ap.add_argument("--skip_step_c", action="store_true")
    args = ap.parse_args()

    logs = load_heldout_logs(pathlib.Path(args.heldout_dir), args.n_campaigns)
    if not logs:
        print(f"No held-out logs found in {args.heldout_dir} — run "
              f"generate_tunable_training_set.py first.")
        return

    if args.af_path:
        af_code = pathlib.Path(args.af_path).read_text()
    else:
        from af_interface_v3 import SEED_PROGRAMS
        af_code = SEED_PROGRAMS["trust_only"]
        print("--af_path not given, using af_interface_v3.SEED_PROGRAMS['trust_only'].")

    sandbox_log_dir = HERE / "evolution_runs" / "diagnostic_v3" / "af_code_logs"

    t0 = time.perf_counter()
    timing = step_a_timing(af_code, logs, args.oracle, sandbox_log_dir=sandbox_log_dir)
    print(f"\nStep A total: {time.perf_counter() - t0:.1f}s for {len(logs)} campaigns "
          f"(mean {timing['mean_campaign_seconds']:.1f}s/campaign)")

    for n_camp in [8, 20, 24]:
        for n_child in [40, 160]:
            hours = (timing["mean_campaign_seconds"] * n_camp * n_child) / 3600
            print(f"  Projected: {n_camp} campaigns x {n_child} children "
                  f"-> {hours:.2f} hours")

    baseline = compute_baseline(logs, args.oracle)

    if not args.skip_step_c:
        step_c_seed_noise(af_code, logs[0], args.oracle, args.n_seed_repeats)

    if args.skip_step_b or not args.population_path:
        return

    population = json.load(open(args.population_path))
    t0 = time.perf_counter()
    step_b_rank_comparison(population, logs, baseline, args.oracle,
                            sandbox_log_dir=sandbox_log_dir)
    print(f"\nStep B total: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
