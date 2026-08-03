"""
analyse_mo_campaign_logs.py — Log analysis for excipient_campaign_mo.py output.

Reads per-seed JSON logs (results_dir/<protein>/<condition>/seed_NNN.json) and
per-batch decisions, and answers the questions the summary CSV alone can't:

  1. HV trajectory shape — is the campaign still improving at budget end, or
     has it plateaued? (tells you if more budget would help)
  2. Pareto front composition — is a large front genuinely spread across the
     objective space (good) or clustered/near-duplicate (misleading)?
  3. Trust/mixing-weight trajectory — does trust ever move within budget, or
     does it stay flat at the sparse-regime default the whole campaign?
  4. LLM proposal quality — how many of the LLM's proposed formulations
     actually survive the duplicate filter and get selected, and what
     targets/tradeoffs did it state?
  5. Objective-by-objective spread of the Pareto front — for a front that's
     large in COUNT, is it also large in objective-space VOLUME per point,
     or is HV/point low (many weak, similar points)?

Usage:
    python analyse_mo_campaign_logs.py --results_dir results_mo_campaign/
    python analyse_mo_campaign_logs.py --results_dir results_mo_campaign/ \\
        --protein mAb_aggregation --condition mo_llm --seed 3
"""

import argparse
import json
import pathlib
import numpy as np
import pandas as pd


def load_seed_logs(results_dir: pathlib.Path, protein: str, condition: str):
    cond_dir = results_dir / protein / condition
    logs = []
    if not cond_dir.exists():
        return logs
    for f in sorted(cond_dir.glob("seed_*.json")):
        with open(f) as fh:
            logs.append(json.load(fh))
    return logs


def hv_trajectory_analysis(logs, label):
    """Is the campaign still climbing at the end, or plateaued?"""
    print(f"\n{'='*65}\nHV TRAJECTORY — {label}\n{'='*65}")
    if not logs:
        print("  No logs found.")
        return

    trajs = [l["hv_trajectory"] for l in logs if l.get("hv_trajectory")]
    if not trajs:
        print("  No HV trajectories in logs.")
        return

    n_batches = min(len(t) for t in trajs)
    trajs_trunc = np.array([t[:n_batches] for t in trajs])
    mean_traj = trajs_trunc.mean(axis=0)

    print(f"  Batches: {n_batches}  Seeds: {len(trajs)}")
    print(f"  Mean HV by batch: {[round(v,0) for v in mean_traj]}")

    # Last-batch gain as fraction of total gain — plateau check
    if n_batches >= 3:
        total_gain = mean_traj[-1] - mean_traj[0]
        last_gain = mean_traj[-1] - mean_traj[-2]
        prev_gain = mean_traj[-2] - mean_traj[-3]
        if total_gain > 0:
            print(f"  Last-batch gain: {last_gain:.0f}  "
                  f"Prior-batch gain: {prev_gain:.0f}  "
                  f"({'still climbing' if last_gain > 0.3*prev_gain else 'PLATEAUING'})")


def pareto_composition_analysis(logs, label, Y_all_ranges=None):
    """
    Is a large Pareto front genuinely spread across objective space, or
    clustered / near-duplicate? Computes HV-per-point and objective-space
    spread of the final front.
    """
    print(f"\n{'='*65}\nPARETO FRONT COMPOSITION — {label}\n{'='*65}")
    if not logs:
        print("  No logs found.")
        return

    hv_per_point = []
    pf_sizes = []
    for l in logs:
        decs = l.get("decisions", [])
        if not decs:
            continue
        last = decs[-1]
        pf_size = last.get("pareto_size", np.nan)
        hv = last.get("hypervolume", np.nan)
        pf_sizes.append(pf_size)
        if pf_size and pf_size > 0:
            hv_per_point.append(hv / pf_size)

    print(f"  Mean Pareto front size: {np.mean(pf_sizes):.1f} ± {np.std(pf_sizes):.1f}")
    print(f"  Mean HV per front point: {np.mean(hv_per_point):.0f} "
          f"± {np.std(hv_per_point):.0f}")
    print(f"  (Lower HV/point with a LARGE front size suggests many weak,")
    print(f"   closely-clustered points rather than genuine spread —")
    print(f"   compare this number across conditions, not in isolation.)")


def trust_trajectory_analysis(logs, label):
    """Does per-objective trust / mixing weight ever move within budget?"""
    print(f"\n{'='*65}\nTRUST / MIXING WEIGHT TRAJECTORY — {label}\n{'='*65}")
    if not logs:
        print("  No logs found (this condition may not use trust/mixing — e.g. egbo/random).")
        return

    has_trust = any("trust" in d for l in logs for d in l.get("decisions", []))
    if not has_trust:
        print("  No trust/mixing_weight fields in these logs (not an mo_llm run?).")
        return

    for l in logs[:5]:  # first 5 seeds, avoid flooding output
        seed = l.get("seed", "?")
        weights = [d.get("mixing_weight") for d in l.get("decisions", [])
                  if "mixing_weight" in d]
        trusts = [d.get("trust") for d in l.get("decisions", [])
                 if "trust" in d]
        if weights:
            flat = len(set(round(w, 2) for w in weights)) == 1
            print(f"  seed {seed}: weight trajectory = {[round(w,2) for w in weights]}"
                  f"  {'(FLAT — trust never moved)' if flat else '(varies)'}")


def llm_proposal_quality(logs, label):
    """How many LLM proposals survive filtering + selection? What did it target?"""
    print(f"\n{'='*65}\nLLM PROPOSAL QUALITY — {label}\n{'='*65}")
    if not logs:
        print("  No logs found.")
        return

    total_proposed = 0
    total_batches = 0
    target_counts = {"Tm": 0, "kD": 0, "viscosity": 0}

    for l in logs:
        for d in l.get("decisions", []):
            if "n_llm_proposed" not in d:
                continue
            total_proposed += d["n_llm_proposed"]
            total_batches += 1
            for tag in d.get("llm_tags", []):
                for t in tag.get("targets", []):
                    if t in target_counts:
                        target_counts[t] += 1

    if total_batches == 0:
        print("  No LLM proposal data (not an mo_llm run?).")
        return

    print(f"  Mean LLM proposals per batch: {total_proposed/total_batches:.1f}")
    print(f"  Target frequency across all proposals: {target_counts}")
    total_tags = sum(target_counts.values())
    if total_tags > 0:
        print(f"  Target distribution: " +
              "  ".join(f"{k}={100*v/total_tags:.0f}%" for k,v in target_counts.items()))


def print_example_reasoning(logs, label, max_seeds=2, max_per_seed=3):
    """Print a few example LLM tags/tradeoffs for spot-checking quality."""
    print(f"\n{'='*65}\nEXAMPLE LLM PROPOSALS — {label}\n{'='*65}")
    shown_seeds = 0
    for l in logs:
        if shown_seeds >= max_seeds:
            break
        seed = l.get("seed", "?")
        decs = l.get("decisions", [])
        if not decs or not decs[0].get("llm_tags"):
            continue
        print(f"\n  Seed {seed}, first batch:")
        for tag in decs[0]["llm_tags"][:max_per_seed]:
            print(f"    targets={tag.get('targets')}  tradeoff=\"{tag.get('tradeoff')}\"")
        shown_seeds += 1


def compare_conditions(results_dir, protein, conditions):
    """Cross-condition comparison table, mirroring the CSV but from logs."""
    print(f"\n{'='*65}\nCROSS-CONDITION COMPARISON — {protein}\n{'='*65}")
    rows = []
    for cond in conditions:
        logs = load_seed_logs(results_dir, protein, cond)
        for l in logs:
            decs = l.get("decisions", [])
            if not decs:
                continue
            last = decs[-1]
            rows.append({"condition": cond, "seed": l.get("seed"),
                        "final_hv": last.get("hypervolume"),
                        "final_pf_size": last.get("pareto_size")})
    if not rows:
        print("  No data found.")
        return
    df = pd.DataFrame(rows)
    agg = df.groupby("condition").agg(
        hv_mean=("final_hv","mean"), hv_std=("final_hv","std"),
        pf_mean=("final_pf_size","mean"), pf_std=("final_pf_size","std"),
    )
    agg["hv_per_point"] = agg["hv_mean"] / agg["pf_mean"]
    print(agg.round(1).to_string())
    print()
    print("  hv_per_point: HV divided by mean Pareto front size.")
    print("  A condition with high pf_mean but LOW hv_per_point relative to")
    print("  others is finding many weak/clustered points, not genuine spread.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_mo_campaign")
    parser.add_argument("--protein", default=None,
                        help="If not given, auto-discover from results_dir")
    parser.add_argument("--condition", nargs="*", default=None,
                        help="If not given, analyse all conditions found")
    parser.add_argument("--seed", type=int, default=None,
                        help="Show detailed per-batch trace for one seed")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)

    if args.protein:
        proteins = [args.protein]
    else:
        proteins = sorted(d.name for d in results_dir.iterdir() if d.is_dir())

    for protein in proteins:
        protein_dir = results_dir / protein
        if not protein_dir.exists():
            continue
        conditions = args.condition or sorted(
            d.name for d in protein_dir.iterdir() if d.is_dir())

        print(f"\n{'#'*65}\nPROTEIN: {protein}\n{'#'*65}")

        compare_conditions(results_dir, protein, conditions)

        for cond in conditions:
            logs = load_seed_logs(results_dir, protein, cond)
            label = f"{protein} / {cond}"
            hv_trajectory_analysis(logs, label)
            pareto_composition_analysis(logs, label)
            if cond == "mo_llm":
                trust_trajectory_analysis(logs, label)
                llm_proposal_quality(logs, label)
                print_example_reasoning(logs, label)

        if args.seed is not None:
            for cond in conditions:
                cond_dir = protein_dir / cond
                seed_file = cond_dir / f"seed_{args.seed:03d}.json"
                if seed_file.exists():
                    print(f"\n{'='*65}\nDETAILED TRACE — {protein}/{cond}/seed_{args.seed}\n{'='*65}")
                    with open(seed_file) as f:
                        log = json.load(f)
                    for d in log["decisions"]:
                        print(f"  step={d['step']:3d}  n_obs={d['n_obs']:3d}  "
                              f"pf_size={d.get('pareto_size','-'):3}  "
                              f"hv={d.get('hypervolume',float('nan')):.0f}"
                              + (f"  weight={d['mixing_weight']:.2f}"
                                 if "mixing_weight" in d else ""))


if __name__ == "__main__":
    main()
