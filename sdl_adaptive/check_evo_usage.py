"""
check_evo_usage.py — Quick check of whether LLM-C used evolutionary_candidates.

Run after a test experiment to verify the utility is being imported and used.

Usage:
    python check_evo_usage.py --results_dir results_evo_test/ --dataset pareto_20210112
"""

import argparse
import pathlib
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_evo_test")
    parser.add_argument("--dataset", default="pareto_20210112")
    parser.add_argument("--condition", default="approach_c_evo",
                        help="Condition name to check (default: approach_c_evo)")
    args = parser.parse_args()

    # Code logs: shared_seed_experiment saves to ./approach_c[_evo]_logs_<dataset>/
    # run_experiment.py saves to ./approach_c_code_logs/
    # Check most-specific path first to avoid reading stale logs
    condition_name = args.condition
    dataset = args.dataset
    logs_path = pathlib.Path(args.results_dir) / dataset / condition_name / "switch_logs.json"

    candidates = [
        pathlib.Path(f"./{condition_name}_logs_{dataset}"),    # shared_seed_experiment
        pathlib.Path(f"./approach_c_code_logs"),               # run_experiment fallback
        pathlib.Path(args.results_dir) / dataset / condition_name / "generated_code",
    ]

    code_dir = None
    for candidate in candidates:
        if candidate.exists() and list(candidate.glob("*.py")):
            code_dir = candidate
            print(f"Reading code from: {code_dir}")
            break

    if code_dir is None:
        print(f"No generated code found. Checked:")
        for c in candidates:
            print(f"  {c}")
        print(f"Ensure controllers/approach_c.py was updated and the run completed.")
        return

    code_files = sorted(code_dir.glob("*.py"))
    if not code_files:
        print(f"No .py files in {code_dir}")
        return

    print(f"Found {len(code_files)} generated code files\n")

    uses_evo  = []
    uses_nov  = []
    uses_gp   = []
    fallbacks = []

    for f in code_files:
        code = f.read_text()
        has_evo = "evolutionary_candidates" in code
        has_nov = "novelty_select" in code
        has_gp  = "GaussianProcess" in code or "gp.predict" in code.lower()
        is_fallback = "np.random.uniform" in code and not has_gp and not has_evo

        uses_evo.append(has_evo)
        uses_nov.append(has_nov)
        uses_gp.append(has_gp)
        fallbacks.append(is_fallback)

        status = []
        if has_evo: status.append("evolutionary_candidates ✓")
        if has_nov: status.append("novelty_select ✓")
        if has_gp:  status.append("GP ✓")
        if is_fallback: status.append("random fallback only")
        if not status:  status.append("other")

        print(f"  {f.name}: {', '.join(status)}")

    n = len(code_files)
    print(f"\nSummary ({n} files):")
    print(f"  Used evolutionary_candidates : {sum(uses_evo)}/{n} ({100*sum(uses_evo)/n:.0f}%)")
    print(f"  Used novelty_select          : {sum(uses_nov)}/{n} ({100*sum(uses_nov)/n:.0f}%)")
    print(f"  Used GP surrogate            : {sum(uses_gp)}/{n} ({100*sum(uses_gp)/n:.0f}%)")
    print(f"  Random fallback only         : {sum(fallbacks)}/{n} ({100*sum(fallbacks)/n:.0f}%)")

    print()
    if sum(uses_evo) == 0:
        print("⚠  LLM did not use evolutionary_candidates at all.")
        print("   Possible causes:")
        print("   1. evolutionary_candidates.py is not in the project root")
        print("   2. system_c.txt was not updated (check prompts/system_c.txt)")
        print("   3. LLM is ignoring the utility — try raising temperature in base.py")
    elif sum(uses_evo) < n // 2:
        print("⚠  LLM used evolutionary_candidates in some but not all seeds.")
        print("   This suggests the prompt is being followed inconsistently.")
        print("   Check whether seeds that used it performed better than those that didn't.")
    else:
        print("✓  LLM consistently used evolutionary_candidates.")
        print("   Ready to run the 80-seed shared experiment.")

    # Check sandbox failures from switch_logs
    if logs_path.exists():
        with open(logs_path) as f:
            logs = json.load(f)
        total_failures = sum(len(s.get("failures", [])) for s in logs)
        total_steps = sum(
            len(s.get("decisions", [])) for s in logs
        )
        print(f"\nSandbox failures: {total_failures} across {len(logs)} seeds")
        if total_failures > 0:
            print("  Some evolutionary_candidates calls may have timed out.")
            print("  If failure rate is high, increase SANDBOX_TIMEOUT in approach_c.py")


if __name__ == "__main__":
    main()
