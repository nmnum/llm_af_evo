"""
evolve_af_2b.py — the actual 2b evolution loop: fitness is real full-
campaign hypervolume from full_replay.run_2b_campaign against a fixed set
of TRAINING campaigns (training_logs/train, disjoint from the held-out set
reserved for final validation via validate_population_2b.py), not the
single-step cached-pool proxy evolve_af.py trains on.

Reuses evolve_af.py's LLM/mock crossover machinery directly (make_child,
tournament_select, SEED_PROGRAMS/SEED_TERM_WEIGHTS, the already-updated
_LLM_SYSTEM_PROMPT with the baseline description / HV-improvement concept
/ fitness semantics / batch-selection clarification) rather than
duplicating it — only the FITNESS function changes (evaluate_af_2b instead
of evaluate_af), same as agreed: this is a fitness fix, not a mechanism
fix.

Fitness: win_rate = fraction of training campaigns where this AF's full-
campaign final HV beats EGBO-novelty's (computed once per training
campaign, shared across every candidate AF — not recomputed per child).
fitness = win_rate - gamma * LOC, matching evolve_af.py's form.

selection_signature here hashes each campaign's realised final HV (rounded)
rather than per-step picked indices (2a's approach) — two AFs producing
the same final HV on every training campaign are behaviourally
indistinguishable for this fitness, so the rank-equivalence dedup still
applies, just measured at the campaign-outcome level instead of the
per-step-pick level.

Known simplification carried over from full_replay.py: stagnant_batches is
not tracked through the replay loop (passed as 0 throughout runtime), so
any AF that conditions on it is scored as if it's always 0 during 2b
training. Not fixed here — same scope note as full_replay.py's docstring.

Usage:
    python evolve_af_2b.py --n_campaigns 8 --pop_size 8 --n_generations 20 \\
        --n_offspring 2 --real_llm --model qwen3-coder:30b \\
        --out_dir evolution_runs/run2_2b
"""

import argparse
import hashlib
import json
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
):
    sys.path.insert(0, str(_p))

from evolve_af import (
    SEED_PROGRAMS, SEED_TERM_WEIGHTS, make_child, tournament_select,
)
from af_interface import count_loc
from full_replay import run_2b_campaign, run_baseline_campaign
from fitness_common import to_allmax

HERE = pathlib.Path(__file__).parent


def load_training_campaigns(train_dir: pathlib.Path, n: int, rng: np.random.Generator) -> list:
    files = sorted(pathlib.Path(train_dir).glob("*.json"))
    if n < len(files):
        idx = rng.choice(len(files), size=n, replace=False)
        files = [files[i] for i in sorted(idx)]
    else:
        files = files[:n]
    return [json.load(open(f)) for f in files]


def compute_baseline_hvs(training_logs: list) -> list:
    print(f"Computing EGBO-novelty baseline on {len(training_logs)} training "
          f"campaigns (shared across every candidate AF this run)...")
    hvs = []
    for i, log in enumerate(training_logs):
        result = run_baseline_campaign(log, seed=i)
        hvs.append(result["final_hv"])
        print(f"  campaign {i}: baseline final_hv={result['final_hv']:.1f}")
    return hvs


def build_pseudo_steps(training_logs: list, baseline_hvs: list, n_sample: int = 3) -> list:
    """
    evolve_af.make_child -> llm_propose_child -> _campaign_diagnostic_text
    expects the 2a "steps" record shape (campaign/step/budget/
    stagnant_batches/front_allmax/egbo_true_gain). Reused here as a
    lightweight adapter so the LLM crossover prompt still shows SOME
    campaign context, without duplicating that function — egbo_true_gain
    is the baseline's campaign-level final_hv here, not a per-step HV
    gain (different scale/meaning), acceptable since this is flavour text
    for the prompt, not something the fitness computation reads.
    """
    n = len(training_logs)
    idx = list(range(0, n, max(1, n // n_sample)))[:n_sample]
    steps = []
    for i in idx:
        log = training_logs[i]
        steps.append({
            "campaign": f"training_campaign_{i}", "step": log["n_init"],
            "budget": log["budget"], "stagnant_batches": 0,
            "front_allmax": to_allmax(np.array(log["Y_init"])),
            "egbo_true_gain": baseline_hvs[i],
        })
    return steps


def evaluate_af_2b(code: str, training_logs: list, baseline_hvs: list,
                    gamma: float, log_dir=None) -> dict:
    """
    Full-campaign fitness. Sandbox failures inside a batch are already
    handled gracefully by run_mo_campaign itself (falls back to random
    candidates for that batch, per its own try/except) — a broken AF just
    produces a bad final_hv on that campaign rather than crashing here, so
    no extra exception handling is needed at this level.

    LOGGING: writes `code` to log_dir ONCE per call, here — NOT via
    sandbox_log_dir threaded down into run_2b_campaign/strategy_evolved_af
    (deliberately left None below). Passing log_dir that deep means
    run_af_in_sandbox writes a NEW file on every one of its calls, and
    strategy_evolved_af calls it once PER BATCH — n_campaigns *
    batches_per_campaign (e.g. 8 * 6 = 48) duplicate writes of the SAME
    candidate's code per evaluate_af_2b call. Same bug evolve_af.py's own
    evaluate_af docstring documents fixing for the 2a path ("wrote the
    same code to a new numbered file per step... 86,400 files from a
    ~17,000-call run") — this file inherited it unfixed from full_replay.py
    until now; run2_2b's already-saved af_code_logs/ predates this fix and
    is NOT retroactively cleaned up (its contents remain valid, just
    heavily duplicated on disk).
    """
    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(log_dir.glob("call_*.py")))
        (log_dir / f"call_{existing:05d}.py").write_text(code)

    wins, hvs = 0, []
    sig_hasher = hashlib.md5()
    for i, log in enumerate(training_logs):
        result = run_2b_campaign(code, log, seed=i, sandbox_log_dir=None)
        hv = result["final_hv"]
        hvs.append(hv)
        sig_hasher.update(str(round(hv, 2)).encode())
        if hv > baseline_hvs[i]:
            wins += 1

    n = len(training_logs)
    win_rate = wins / n if n else 0.0
    loc = count_loc(code)
    fitness = win_rate - gamma * loc
    return {"win_rate": win_rate, "loc": loc, "fitness": fitness,
            "mean_hv": float(np.mean(hvs)) if hvs else float("nan"),
            "n_campaigns": n, "selection_signature": sig_hasher.hexdigest()}


def run_evolution_2b(training_logs: list, baseline_hvs: list, pop_size: int,
                      n_generations: int, n_offspring: int, gamma: float,
                      mock: bool, model: str, seed: int, extra_seeds: dict,
                      log_dir: pathlib.Path = None) -> dict:
    rng = np.random.default_rng(seed)
    pseudo_steps = build_pseudo_steps(training_logs, baseline_hvs)

    seeds = dict(SEED_PROGRAMS)
    seeds.update(extra_seeds)
    seed_term_weights = dict(SEED_TERM_WEIGHTS)  # extra_seeds have no term_weights entry

    population = []
    for name, code in seeds.items():
        result = evaluate_af_2b(code, training_logs, baseline_hvs, gamma, log_dir=log_dir)
        population.append({"id": name, "code": code,
                            "term_weights": seed_term_weights.get(name), **result})
    population = sorted(population, key=lambda p: -p["fitness"])[:max(pop_size, len(population))]

    history = [{"generation": 0, "best_fitness": population[0]["fitness"],
                "best_win_rate": population[0]["win_rate"],
                "best_mean_hv": population[0]["mean_hv"]}]
    print(f"gen 0: best fitness={population[0]['fitness']:.4f} "
          f"win_rate={population[0]['win_rate']:.3f} mean_hv={population[0]['mean_hv']:.1f} "
          f"({population[0]['id']})")

    n_llm_calls, n_llm_failures = 0, 0
    for gen in range(1, n_generations + 1):
        # best_so_far is now a required make_child argument (evolve_af.py
        # added it for the explainability-line/best-so-far-docstring
        # feature) — this call site was missed when that change landed,
        # which would have made every real-LLM generation here raise
        # TypeError (wrong argument count/positions). Fixed by computing
        # it the same way evolve_af.py's own run_evolution does.
        best_so_far = max(population, key=lambda p: p["fitness"])
        children = []
        for i in range(n_offspring):
            parent_a = tournament_select(population, rng)
            parent_b = tournament_select(population, rng)
            code, term_weights, used_llm = make_child(parent_a, parent_b, best_so_far,
                                                        pseudo_steps, mock, model, rng)
            if not mock:
                n_llm_calls += 1
                n_llm_failures += (not used_llm)
            result = evaluate_af_2b(code, training_logs, baseline_hvs, gamma, log_dir=log_dir)
            children.append({"id": f"gen{gen}_child{i}", "code": code,
                              "term_weights": term_weights, "used_llm": used_llm, **result})

        existing_sigs = {p["selection_signature"] for p in population}
        novel_children, n_duplicate = [], 0
        for c in children:
            if c["selection_signature"] in existing_sigs:
                n_duplicate += 1
                continue
            existing_sigs.add(c["selection_signature"])
            novel_children.append(c)

        population = sorted(population + novel_children, key=lambda p: -p["fitness"])[:pop_size]
        history.append({"generation": gen, "best_fitness": population[0]["fitness"],
                         "best_win_rate": population[0]["win_rate"],
                         "best_mean_hv": population[0]["mean_hv"],
                         "n_rank_equivalent_duplicates": n_duplicate})
        print(f"gen {gen}: best fitness={population[0]['fitness']:.4f} "
              f"win_rate={population[0]['win_rate']:.3f} mean_hv={population[0]['mean_hv']:.1f} "
              f"({population[0]['id']})  duplicates={n_duplicate}/{len(children)}")

    if not mock:
        print(f"\nReal-LLM calls: {n_llm_calls}, fell back to mock crossover: "
              f"{n_llm_failures} ({100 * n_llm_failures / max(1, n_llm_calls):.0f}%)")

    return {"population": population, "history": history,
            "n_llm_calls": n_llm_calls, "n_llm_failures": n_llm_failures}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default=str(HERE / "training_logs" / "train"))
    ap.add_argument("--n_campaigns", type=int, default=8)
    ap.add_argument("--pop_size", type=int, default=8)
    ap.add_argument("--n_generations", type=int, default=20)
    ap.add_argument("--n_offspring", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=0.005)
    ap.add_argument("--mock", action="store_true", default=True)
    ap.add_argument("--real_llm", dest="mock", action="store_false")
    ap.add_argument("--model", default="qwen3-coder:30b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra_seed_run1_id", default="gen5_child7",
                     help="Pull this AF's code from run1's final_population.json "
                          "as an additional seed (empirically validated direction, "
                          "even though it lost the 20-campaign gate test).")
    ap.add_argument("--run1_population_path",
                     default=str(HERE / "evolution_runs" / "run1" / "final_population.json"))
    ap.add_argument("--out_dir", default=str(HERE / "evolution_runs" / "run2_2b"))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_log_dir = out_dir / "af_code_logs"

    rng = np.random.default_rng(args.seed)
    training_logs = load_training_campaigns(pathlib.Path(args.train_dir), args.n_campaigns, rng)
    print(f"Loaded {len(training_logs)} training campaigns.")

    extra_seeds = {}
    if args.extra_seed_run1_id:
        run1_pop = json.load(open(args.run1_population_path))
        match = next((p for p in run1_pop if p["id"] == args.extra_seed_run1_id), None)
        if match:
            extra_seeds[args.extra_seed_run1_id] = match["code"]
            print(f"Seeded with {args.extra_seed_run1_id} from {args.run1_population_path}")
        else:
            print(f"WARNING: {args.extra_seed_run1_id!r} not found in "
                  f"{args.run1_population_path}, skipping.")

    baseline_hvs = compute_baseline_hvs(training_logs)

    result = run_evolution_2b(
        training_logs, baseline_hvs, args.pop_size, args.n_generations, args.n_offspring,
        args.gamma, args.mock, args.model, args.seed, extra_seeds, log_dir=code_log_dir,
    )

    best = result["population"][0]
    print(f"\nBest AF: fitness={best['fitness']:.4f} win_rate={best['win_rate']:.3f} "
          f"mean_hv={best['mean_hv']:.1f} LOC={best['loc']} ({best['id']})")
    print(f"\n{best['code']}")

    with open(out_dir / "best_af.py", "w") as f:
        f.write(best["code"])
    with open(out_dir / "history.json", "w") as f:
        json.dump({"history": result["history"], "mock": args.mock,
                    "n_llm_calls": result["n_llm_calls"],
                    "n_llm_failures": result["n_llm_failures"],
                    "baseline_hvs": baseline_hvs}, f, indent=2)
    with open(out_dir / "final_population.json", "w") as f:
        json.dump([{k: v for k, v in p.items() if k != "term_weights"}
                    for p in result["population"]], f, indent=2)

    print(f"\nSaved best AF, history, and final population to {out_dir}")
    print(f"Next: python validate_population_2b.py --population_path "
          f"{out_dir / 'final_population.json'}")


if __name__ == "__main__":
    main()
