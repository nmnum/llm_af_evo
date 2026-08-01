"""
diagnose_win_overlap.py — resolves the "byte-identical held-out results"
puzzle from run_v2_coatings_gamma001_v2/final_population.json: six
structurally-different, nonzero-uncertainty-weight AFs (gen6_child0,
hint_fixed_ucb, gen4_child0, gen1_child1, gen9_child0, gen5_child0) all
land on the EXACT SAME win_rate (0.6933333333333334 == 52/75), while
trust_only and gen10_child1 both land on a different shared value (0.68 ==
51/75) -- but all 8 have DIFFERENT selection_signature hashes, meaning
their actual per-campaign HV trajectories (and therefore likely their
per-step picks, consistent with diagnose_mu_sigma_dominance.py's sweep
showing sigma genuinely moves top-k membership at these AFs' real
weights) are NOT identical.

Two live explanations for identical win_rate + different signatures:
  (a) campaign-difficulty effect: these AFs win/lose on the SAME subset
      of campaigns (same 52 of 75), because campaign difficulty dominates
      AF choice at this budget/domain -- different HV values, same
      win/loss pattern.
  (b) coincidental collision: win/loss patterns differ (different sets of
      52 campaigns won), landing on the same COUNT by chance -- plausible
      at n=75 campaigns, where win_rate only has 76 possible values, so
      several similarly-effective-but-different AFs colliding on a count
      is not statistically surprising on its own.

This script re-runs each candidate's ACTUAL saved source code (from
final_population.json, not an idealized formula) via
full_replay.run_2b_campaign against the same 75 training campaigns/
baseline_hvs the original run used, and reports the per-campaign win/loss
vector for each, plus pairwise overlap of which SPECIFIC campaigns were
won -- directly distinguishing (a) from (b).

Cost note: this is 8 candidates x 75 campaigns = 600 full sequential BO
campaign replays (GP fits per batch) -- comparable to one evaluate_af_2b
pass per already-known candidate, NOT a re-run of evolution (no LLM
calls, no generations, no population). Still not free; consider
--candidates to restrict to a subset first if you want a quick look.

Usage: python diagnose_win_overlap.py [run_dir] [--train_dir DIR] [--candidates id1 id2 ...]
"""

import argparse
import json
import pathlib

import numpy as np

from full_replay import run_2b_campaign


def load_run(run_dir: pathlib.Path):
    with open(run_dir / "final_population.json") as f:
        population = json.load(f)
    with open(run_dir / "history.json") as f:
        history = json.load(f)
    return population, history["baseline_hvs"]


def load_training_logs(train_dir: pathlib.Path, n: int) -> list:
    """Same determinism as evolve_af_v2.load_training_campaigns when n >=
    len(files): sorted glob, first n -- no rng needed since the original
    coatings run used n_campaigns == the full 75-file training set."""
    files = sorted(train_dir.glob("*.json"))[:n]
    return [json.load(open(f)) for f in files]


def per_campaign_wins(code: str, training_logs: list, baseline_hvs: list,
                       oracle_family: str) -> np.ndarray:
    wins = np.zeros(len(training_logs), dtype=bool)
    for i, log in enumerate(training_logs):
        result = run_2b_campaign(code, log, seed=i, sandbox_log_dir=None,
                                  oracle_family=oracle_family)
        hv = result["hv_trajectory"][-1] if result["hv_trajectory"] else float("-inf")
        wins[i] = hv > baseline_hvs[i]
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?",
                     default="evolution_runs/run_v2_coatings_gamma001_v2")
    ap.add_argument("--train_dir", default="training_logs_coatings_100/train")
    ap.add_argument("--oracle-family", default="coatings")
    ap.add_argument("--candidates", nargs="*", default=None,
                     help="Restrict to these candidate ids (default: all in final_population.json)")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    population, baseline_hvs = load_run(run_dir)
    if args.candidates:
        population = [p for p in population if p["id"] in args.candidates]

    training_logs = load_training_logs(pathlib.Path(args.train_dir), len(baseline_hvs))
    assert len(training_logs) == len(baseline_hvs), (
        f"training log count ({len(training_logs)}) != baseline_hvs count "
        f"({len(baseline_hvs)}) -- wrong --train_dir for this run_dir?")

    print(f"Re-running {len(population)} candidates x {len(training_logs)} campaigns "
          f"(oracle_family={args.oracle_family})...\n")

    win_vectors = {}
    for p in population:
        print(f"  {p['id']} (saved win_rate={p['win_rate']:.4f}, "
              f"selection_signature={p['selection_signature'][:12]}...)")
        wins = per_campaign_wins(p["code"], training_logs, baseline_hvs, args.oracle_family)
        win_vectors[p["id"]] = wins
        recomputed_rate = wins.mean()
        match = "OK" if abs(recomputed_rate - p["win_rate"]) < 1e-9 else "MISMATCH"
        print(f"    recomputed win_rate={recomputed_rate:.4f} [{match} vs. saved]")

    ids = list(win_vectors.keys())
    print(f"\nPairwise winning-campaign-SET overlap (Jaccard: |A∩B| / |A∪B|; "
          f"1.0 = identical set of campaigns won, not just identical count):")
    header = "".join(f"{i:>14s}" for i in ids)
    print(f"{'':20s}{header}")
    for a in ids:
        row = []
        for b in ids:
            set_a, set_b = set(np.where(win_vectors[a])[0]), set(np.where(win_vectors[b])[0])
            union = set_a | set_b
            jaccard = len(set_a & set_b) / len(union) if union else 1.0
            row.append(f"{jaccard:14.3f}")
        print(f"{a:20s}{''.join(row)}")

    print("""
Reading this:
  Jaccard == 1.0 between two same-win_rate candidates -> they win the
    EXACT SAME campaigns, not just the same count. That's explanation (a):
    campaign difficulty dominates which AF wins/loses at this budget, and
    the different selection_signatures just reflect different HV VALUES
    on those same wins/losses, not different win/loss patterns.
  Jaccard meaningfully < 1.0 despite identical win_rate -> explanation
    (b): different campaigns won, same count by coincidence -- at n=75
    campaigns (76 possible win_rate values), this is not a red flag on
    its own, just a coarse-metric collision. If this is the case, prefer
    mean_margin/fitness (continuous) over win_rate for distinguishing
    these candidates rather than reading the tied win_rate as meaningful.
""")


if __name__ == "__main__":
    main()
