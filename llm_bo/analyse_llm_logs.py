"""
analyse_llm_logs.py — Post-hoc analysis of LLM-BO decision logs.
Reads ONLY from per-seed JSON logs. No CSV dependency.

Usage:
    python analyse_llm_logs.py --results_dir results_posthoc_llm_v3/
    python analyse_llm_logs.py --results_dir results_posthoc_llm_v3/ \
        --dataset pareto_20210112 coatings hartmann6 hartmann3 pareto_20201218
"""

import argparse
import json
import pathlib
import numpy as np
from scipy import stats


def discover_datasets(results_dir: pathlib.Path) -> list[str]:
    """Find all datasets that have at least one seed log."""
    datasets = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and any(d.glob("*/seed_*.json")):
            datasets.append(d.name)
    return datasets


def discover_conditions(dataset_dir: pathlib.Path) -> list[str]:
    """Find all conditions that have seed logs for this dataset."""
    conds = []
    for d in sorted(dataset_dir.iterdir()):
        if d.is_dir() and any(d.glob("seed_*.json")):
            conds.append(d.name)
    return conds


def load_seed_logs(dataset_dir: pathlib.Path, condition: str) -> list[dict]:
    cond_dir = dataset_dir / condition
    logs = []
    for f in sorted(cond_dir.glob("seed_*.json")):
        with open(f) as fh:
            logs.append(json.load(fh))
    return logs


def all_queried_y(log: dict) -> list[float]:
    """Flatten all queried_y values from a seed log's decisions."""
    ys = []
    for dec in log.get("decisions", []):
        ys.extend(dec.get("queried_y", []))
    return ys


def paired_ttest(a: list, b: list) -> tuple[float, float]:
    a, b = np.array(a), np.array(b)
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    diff = a - b
    _, p = stats.ttest_1samp(diff, 0)
    return float(np.mean(diff)), float(p)


def steps_to_threshold(decisions: list, y_all: list, threshold: float) -> int | None:
    target = max(y_all) * threshold
    running_max = -np.inf
    for dec in decisions:
        for y in dec.get("queried_y", []):
            running_max = max(running_max, y)
            if running_max >= target:
                return dec["step"]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_posthoc_llm_v3")
    parser.add_argument("--dataset", "--datasets", nargs="*", default=None,
                        dest="dataset")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--reasoning", action="store_true",
                        help="Print LLM reasoning traces after analysis")
    parser.add_argument("--reasoning_seeds", type=int, default=3)
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)

    # Discover datasets purely from directory structure
    all_available = discover_datasets(results_dir)
    if not all_available:
        print(f"No seed_*.json logs found under {results_dir}")
        return

    if args.dataset:
        requested = args.dataset
        datasets = [d for d in requested if (results_dir / d).is_dir()
                    and any((results_dir / d).glob("*/seed_*.json"))]
        missing = [d for d in requested if d not in datasets]
        if missing:
            print(f"Warning: datasets not found in logs: {missing}")
            print(f"Available: {all_available}")
    else:
        datasets = all_available

    print(f"Analysing datasets: {datasets}")
    print("="*70)
    print("LLM-BO LOG ANALYSIS")
    print("="*70)

    # Collect AUC per condition per dataset for summary table
    summary: dict[str, dict[str, list[float]]] = {}

    for ds in datasets:
        ds_dir = results_dir / ds
        conditions = discover_conditions(ds_dir)
        if not conditions:
            print(f"\nNo conditions found for {ds}, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"Dataset: {ds}  (conditions: {conditions})")
        print(f"{'='*70}")

        # Load all logs
        all_logs: dict[str, list[dict]] = {
            c: load_seed_logs(ds_dir, c) for c in conditions
        }
        n_seeds = max(len(v) for v in all_logs.values())

        # Collect AUCs from logs (not CSV)
        summary[ds] = {}
        for cond, logs in all_logs.items():
            aucs = [log.get("auc", float("nan")) for log in logs]
            summary[ds][cond] = aucs

        # ── 1. Convergence speed ─────────────────────────────────────────────
        print(f"\n1. CONVERGENCE SPEED (steps to {args.threshold:.0%} of campaign best)")
        for cond, logs in all_logs.items():
            step_list = []
            never = 0
            for log in logs:
                decs = log.get("decisions", [])
                y_all = all_queried_y(log)
                if not y_all:
                    never += 1
                    continue
                s = steps_to_threshold(decs, y_all, args.threshold)
                if s is None:
                    never += 1
                    step_list.append(log.get("budget", 50))
                else:
                    step_list.append(s)
            if step_list:
                print(f"   {cond:<14} median={np.median(step_list):.0f}  "
                      f"mean={np.mean(step_list):.1f}±{np.std(step_list):.1f}  "
                      f"never_reached={never}/{len(logs)}")

        # ── 2. Early vs late gain ────────────────────────────────────────────
        print(f"\n2. EARLY vs LATE GAIN (first 50% / second 50% of budget)")
        for cond, logs in all_logs.items():
            early_bests, late_gains = [], []
            for log in logs:
                decs = log.get("decisions", [])
                budget = log.get("budget", 50)
                mid = budget // 2
                y_all = all_queried_y(log)
                if not y_all:
                    continue
                gmax = max(y_all); gmin = min(y_all); rng = gmax - gmin + 1e-12
                early_y = [y for d in decs for y in d.get("queried_y", [])
                            if d["step"] <= mid]
                if early_y:
                    eb = (max(early_y) - gmin) / rng
                    lb = (gmax - max(early_y)) / rng
                    early_bests.append(eb)
                    late_gains.append(max(0, lb))
            if early_bests:
                print(f"   {cond:<14} early_best={np.mean(early_bests):.3f}  "
                      f"late_gain={np.mean(late_gains):.3f}")

        # ── 3. Seed-level: primary LLM cond vs EGBO ─────────────────────────
        llm_cond = next((c for c in ["llm_nopad", "llm_bo"] if c in all_logs), None)
        if llm_cond and "egbo" in all_logs:
            print(f"\n3. SEED-LEVEL: {llm_cond} vs EGBO")
            llm_logs  = {log["seed"]: log for log in all_logs[llm_cond]}
            egbo_logs = {log["seed"]: log for log in all_logs["egbo"]}
            common = sorted(set(llm_logs) & set(egbo_logs))

            early_wins, late_wins, egbo_wins, tied = 0, 0, 0, 0
            for seed in common:
                ll, el = llm_logs[seed], egbo_logs[seed]
                la, ea = ll.get("auc", 0), el.get("auc", 0)
                budget = ll.get("budget", 50)
                mid = budget // 2
                if la > ea + 0.01:
                    ll_early = [y for d in ll["decisions"]
                                for y in d.get("queried_y", [])
                                if d["step"] <= mid]
                    el_early = [y for d in el["decisions"]
                                for y in d.get("queried_y", [])
                                if d["step"] <= mid]
                    if ll_early and el_early and max(ll_early) > max(el_early):
                        early_wins += 1
                    else:
                        late_wins += 1
                elif ea > la + 0.01:
                    egbo_wins += 1
                else:
                    tied += 1

            total = len(common)
            print(f"   {llm_cond} wins early: {early_wins}/{total}")
            print(f"   {llm_cond} wins late:  {late_wins}/{total}")
            print(f"   EGBO wins:           {egbo_wins}/{total}")
            print(f"   Tied:                {tied}/{total}")
            if early_wins > late_wins:
                print(f"   → LLM advantage is EARLY-CAMPAIGN")
            else:
                print(f"   → LLM advantage is LATE-CAMPAIGN")

        # ── 4. Improvement by third ──────────────────────────────────────────
        print(f"\n4. IMPROVEMENT RATE BY CAMPAIGN THIRD")
        for cond, logs in all_logs.items():
            thirds = [[], [], []]
            for log in logs:
                decs = log.get("decisions", [])
                budget = log.get("budget", 50)
                y_all = all_queried_y(log)
                if not y_all:
                    continue
                gmax = max(y_all); gmin = min(y_all); rng = gmax - gmin + 1e-12
                prev = gmin
                for i, (lo_f, hi_f) in enumerate(
                        [(0, 0.33), (0.33, 0.66), (0.66, 1.0)]):
                    ty = [y for d in decs for y in d.get("queried_y", [])
                          if lo_f * budget <= d["step"] < hi_f * budget]
                    if ty:
                        gain = (max(ty) - prev) / rng
                        thirds[i].append(max(0.0, gain))
                        prev = max(prev, max(ty))
            labels = ["early(0-33%)", "mid(33-66%)", "late(66-100%)"]
            row = f"   {cond:<14}"
            for g, lab in zip(thirds, labels):
                row += f"  {lab}:{np.mean(g):.3f}" if g else f"  {lab}:n/a"
            print(row)

    # ── Cross-dataset summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CROSS-DATASET SUMMARY (paired t-tests, AUC from logs)")
    print(f"{'='*70}")
    all_conds = sorted({c for ds_conds in summary.values() for c in ds_conds})
    print(f"\n{'Dataset':<22}", end="")
    for c in all_conds:
        print(f"  {c[:10]:<10}", end="")
    print()
    print("-"*70)
    for ds, conds in summary.items():
        print(f"  {ds:<20}", end="")
        for c in all_conds:
            vals = conds.get(c, [])
            m = np.nanmean(vals) if vals else float("nan")
            print(f"  {m:.3f}     ", end="")
        print()

    print()
    print("Paired t-tests (primary LLM vs others):")
    for ds, conds in summary.items():
        llm_key = next((k for k in ["llm_nopad","llm_bo"] if k in conds), None)
        if not llm_key:
            continue
        print(f"\n  {ds}:")
        for other in all_conds:
            if other == llm_key or other not in conds:
                continue
            d, p = paired_ttest(conds[llm_key], conds[other])
            sig = " *" if not np.isnan(p) and p < 0.05 else ""
            print(f"    {llm_key} vs {other:<14} Δ={d:+.3f}  p={p:.4f}{sig}")


def print_llm_reasoning(results_dir, dataset, condition="llm_nopad",
                         max_seeds=3, steps_per_seed=3):
    """Print LLM reasoning traces from per-seed JSON logs.
    Shows the free-text explanation the LLM gave for each batch decision.
    Useful for explaining behaviour on specific datasets (e.g. flat landscape).
    """
    cond_dir = results_dir / dataset / condition
    logs = sorted(cond_dir.glob("seed_*.json"))[:max_seeds] if cond_dir.exists() else []
    if not logs:
        print(f"  No logs found: {cond_dir}")
        return
    print(f"\nLLM REASONING TRACES  {dataset} / {condition}")
    print("="*70)
    for log_path in logs:
        with open(log_path) as f:
            log = json.load(f)
        seed = log.get("seed", "?")
        auc  = log.get("auc", float("nan"))
        decs = log.get("decisions", [])
        print(f"\nSeed {seed}  auc={auc:.3f}")
        shown = 0
        for dec in decs:
            r = dec.get("llm_reasoning", "").strip()
            if not r:
                continue
            step, n_obs = dec.get("step", "?"), dec.get("n_obs", "?")
            print(f"  Step {step} (n_obs={n_obs}):")
            words, line, out = r.split(), [], []
            for w in words:
                if len(" ".join(line + [w])) > 65:
                    out.append("    " + " ".join(line)); line = [w]
                else:
                    line.append(w)
            if line: out.append("    " + " ".join(line))
            print("\n".join(out))
            shown += 1
            if shown >= steps_per_seed:
                break
        if shown == 0:
            print("  (no reasoning traces — re-run with updated posthoc script)")


    # Print reasoning traces if requested
    if args.reasoning:
        for ds in datasets:
            for cond in ["llm_nopad", "llm_bo"]:
                if (results_dir / ds / cond).exists():
                    print_llm_reasoning(
                        results_dir, ds, cond,
                        max_seeds=args.reasoning_seeds)
                    break

if __name__ == "__main__":
    main()
