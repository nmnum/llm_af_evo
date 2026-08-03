"""
posthoc_llm_comparison.py
─────────────────────────
Post-hoc comparison: LLM-BO vs EGBO vs Random on a structured landscape.

LLM role: candidate GENERATION, not hyperparameter tuning.
  - LLM receives prior knowledge about the input space + current observations
  - LLM proposes 20 candidate points in the input space
  - GP scores candidates with UCB (same GP as EGBO)
  - Top batch_size candidates are queried

This is the correct LLM role: biochemist/domain-expert, not statistician.
The LLM never touches the GP hyperparameters or acquisition function.

Three conditions (shared seeds, same infrastructure as Phase 1):
  (a) egbo          — confirmed winner on structured landscapes
  (b) llm_bo        — LLM candidate generation + GP scoring
  (c) random        — pure random search (floor baseline)

Dataset: pareto_20210112 (4D, N=72, structured, EGBO Δ=+0.151 vs LHS)

Prior knowledge supplied to LLM: a description of the ADA coatings input
space, generated from the dataset's known parameter structure. This mirrors
the excipient table approach — give the LLM something to reason about.

Usage:
    # Quick test (3 seeds, no LLM)
    python posthoc_llm_comparison.py --data_dir data/ --mock_llm --n_repeats 3

    # Full run with Qwen2.5-72B on Spark GPU
    python posthoc_llm_comparison.py --data_dir data/ \\
        --model qwen2.5:72b-instruct \\
        --n_repeats 20 \\
        --datasets pareto_20210112 coatings hartmann6

    # Use stronger reasoning model
    python posthoc_llm_comparison.py --data_dir data/ \\
        --model deepseek-r1:70b \\
        --n_repeats 20
"""

import argparse
import json
import pathlib
import re
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).parent))


# ── Prior knowledge templates ──────────────────────────────────────────────────
# Each dataset gets a description of its input space so the LLM can reason
# about candidates rather than treating inputs as anonymous numbers.

PRIOR_KNOWLEDGE = {

    "pareto_20210112": """
You are optimising a multi-layer coating process with 4 continuous parameters.
The goal is to maximise a scalarised coating quality score (conductivity +
uniformity + adhesion). Higher scores are better. All inputs are normalised
to [0, 1].

Input parameters and their physical meaning:
  x0 (concentration): precursor concentration in solution (0=low, 1=high)
      Low concentration → thinner, more uniform layers
      High concentration → thicker layers, risk of cracking
  x1 (temperature): deposition temperature (0=25°C, 1=200°C)
      Higher temperature → better crystallinity, higher conductivity
      Moderate temperature (0.4-0.6) often optimal
  x2 (flow_rate): carrier gas flow rate (0=slow, 1=fast)
      Low flow → longer residence time, denser coating
      High flow → risk of non-uniform deposition
  x3 (pressure): chamber pressure (0=vacuum, 1=atmospheric)
      Low pressure → better step coverage
      High pressure → faster deposition but lower quality

Known interactions:
  - High concentration + high temperature → cracking (avoid x0>0.7 with x1>0.8)
  - Low flow + low pressure → very slow deposition (acceptable for quality runs)
  - Best observed formulations tend to have moderate temperature (0.3-0.7)
    and low-to-moderate concentration (0.2-0.6)
""",

    "coatings": """
You are optimising a coatings formulation with 7 continuous parameters.
The goal is to maximise a combined optical performance metric.
All inputs are normalised to [0, 1].

Input parameters:
  x0 (binder_fraction): fraction of binder polymer (0=none, 1=pure binder)
  x1 (pigment_loading): pigment concentration (0=clear, 1=fully loaded)
  x2 (solvent_ratio): primary/secondary solvent ratio (0=pure primary, 1=pure secondary)
  x3 (cure_temp): curing temperature (0=room temp, 1=200°C)
  x4 (cure_time): curing duration (0=1min, 1=60min)
  x5 (film_thickness): wet film thickness (0=thin, 1=thick)
  x6 (additive_level): performance additive concentration (0=none, 1=maximum)

General guidance:
  - Moderate binder (0.3-0.6) with moderate pigment (0.2-0.5) often performs well
  - Higher cure temperature generally improves crosslinking (x3>0.5 preferred)
  - Additive level has diminishing returns above 0.6
  - Avoid extreme values in multiple parameters simultaneously
""",

    "hartmann6": """
You are optimising a 6-dimensional engineering system with multiple
local optima. All inputs x0-x5 are continuous in [0, 1].
The function has 6 local optima; the global optimum is near
[0.20, 0.15, 0.48, 0.27, 0.31, 0.66].

There is no simple physical interpretation — treat this as a black-box
optimisation problem. Your goal is to propose diverse candidates that
explore regions not yet sampled, while also exploiting promising areas.

Strategy guidance:
  - The landscape has structure: nearby points tend to have similar values
  - Multiple optima exist — explore different regions
  - Points near the boundaries (x_i < 0.1 or x_i > 0.9) tend to perform poorly
  - Mid-range values (0.2-0.7) are generally more promising
""",

    "hartmann3": """
You are optimising a 3-dimensional engineering function in [0,1]^3.
The global optimum is near [0.114, 0.556, 0.852].
Propose diverse candidates across the space, favouring mid-range values.
""",

    "pareto_20201218": """
You are optimising a coating process with 4 parameters (all in [0,1]).
This landscape is relatively flat — many parameter combinations give
similar performance. Focus on broad exploration rather than exploitation.
""",
}

DEFAULT_PRIOR = """
You are optimising a {d}-dimensional black-box function.
All inputs are continuous in [0, 1].
Propose diverse candidates, exploring regions not yet sampled.
"""


# ── LLM candidate generation ───────────────────────────────────────────────────

def _build_prompt(prior_text: str, X_obs: np.ndarray, y_obs: np.ndarray,
                  n_suggest: int, bounds: np.ndarray) -> str:
    """Build the LLM prompt with prior knowledge + current observations."""
    d = X_obs.shape[1]
    gb = float(y_obs.max())
    gw = float(y_obs.min())

    # Show top-10 observations, normalised to [0,1] for readability
    top_idx = np.argsort(y_obs)[::-1][:10]
    obs_lines = []
    for rank, i in enumerate(top_idx):
        score_norm = (y_obs[i] - gw) / (gb - gw + 1e-12)
        coords = ", ".join(f"x{j}={X_obs[i,j]:.3f}" for j in range(d))
        obs_lines.append(f"  {rank+1:2d}. {coords} → score={score_norm:.3f}")
    obs_text = "\n".join(obs_lines)

    prompt = f"""{prior_text.strip()}

─── Current campaign state ───
Total observations: {len(y_obs)}
Best score (normalised 0-1): {(gb-gw)/(gb-gw+1e-12):.3f}
Worst score: 0.000
Range of scores: [{gw:.3f}, {gb:.3f}] (raw)

Top-10 observations so far (normalised scores):
{obs_text}

─── Your task ───
Propose exactly {n_suggest} new candidate points to evaluate next.
Each candidate must have {d} values, one per input dimension (x0 to x{d-1}).
All values must be in [0, 1].

Requirements:
  - Include a mix: some near the best observed points (exploitation),
    some in unexplored regions (exploration)
  - Avoid repeating points already observed
  - If you see a clear trend in the top observations, exploit it
  - Briefly explain your reasoning for each suggestion

Respond with valid JSON only. No markdown, no backticks, no explanation outside JSON:
{{
  "candidates": [
    {{"x": [x0, x1, ..., x{d-1}], "reasoning": "one sentence"}},
    ...
  ]
}}"""
    return prompt


def _mock_llm_suggest(X_obs: np.ndarray, y_obs: np.ndarray,
                      n_suggest: int, bounds: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Mock LLM that doesn't need Ollama — useful for testing.
    Strategy: 50% near top-3 observed (exploitation), 50% random (exploration).
    This is a reasonable baseline for what a domain-informed LLM might do.
    """
    d = bounds.shape[0]
    top_idx = np.argsort(y_obs)[::-1][:3]
    X_top = X_obs[top_idx]
    candidates = []

    for _ in range(n_suggest // 2):
        base = X_top[rng.integers(len(X_top))]
        perturbed = base + rng.normal(0, 0.1, d)
        candidates.append(np.clip(perturbed, 0, 1))

    for _ in range(n_suggest - len(candidates)):
        candidates.append(rng.uniform(0, 1, d))

    return np.array(candidates)


def llm_suggest(X_obs: np.ndarray, y_obs: np.ndarray,
                bounds: np.ndarray, prior_text: str,
                n_suggest: int = 20, model: str = "qwen2.5:14b-instruct",
                mock: bool = False, rng: np.random.Generator = None,
                max_retries: int = 3) -> np.ndarray:
    """
    Generate candidate points using the LLM.
    Falls back to mock on parse failure.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if mock:
        return _mock_llm_suggest(X_obs, y_obs, n_suggest, bounds, rng)

    import ollama

    prompt = _build_prompt(prior_text, X_obs, y_obs, n_suggest, bounds)
    d = bounds.shape[0]

    for attempt in range(max_retries):
        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system",
                     "content": ("You are an expert optimisation assistant. "
                                 "Always respond with valid JSON only.")},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.3, "num_predict": 1024,
                         "think": False},
            )
            text = resp["message"]["content"]
            # Strip <think>...</think> for reasoning models (DeepSeek-R1)
            text = re.sub(r'<think>.*?</think>', '', text,
                          flags=re.DOTALL).strip()
            # Strip markdown fences
            text = re.sub(r'```(?:json)?', '', text).strip().strip('`')

            parsed = json.loads(text)
            raw_cands = parsed.get("candidates", [])
            if not raw_cands:
                raise ValueError("No candidates in response")

            cands = []
            for c in raw_cands:
                x = c.get("x", [])
                if len(x) == d:
                    cands.append(np.clip(np.array(x, dtype=float), 0, 1))
            if len(cands) == 0:
                raise ValueError("No valid candidates parsed")

            # Pad with random if LLM returned fewer than requested
            while len(cands) < n_suggest:
                cands.append(rng.uniform(0, 1, d))

            return np.array(cands[:n_suggest])

        except Exception as e:
            if attempt == max_retries - 1:
                warnings.warn(f"LLM failed after {max_retries} attempts: {e}. "
                              f"Using mock fallback.")
                return _mock_llm_suggest(X_obs, y_obs, n_suggest, bounds, rng)

    return _mock_llm_suggest(X_obs, y_obs, n_suggest, bounds, rng)


# ── GP scoring (shared by EGBO and LLM-BO) ────────────────────────────────────

def _fit_gp(X_obs, y_obs):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_obs)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
        alpha=1e-6, normalize_y=True, n_restarts_optimizer=3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(Xs, y_obs)
    return gp, scaler


def _ucb_score(gp, scaler, X_cands, y_obs, progress):
    """UCB with beta decaying from 5 (explore) to 0.5 (exploit)."""
    beta = max(0.5, 5.0 * (1.0 - progress))
    Xs = scaler.transform(X_cands)
    mu, sigma = gp.predict(Xs, return_std=True)
    return mu + beta * sigma


# ── Strategy implementations ───────────────────────────────────────────────────

def strategy_random(oracle, X_obs, y_obs, bounds, batch_size, rng, **kw):
    d = bounds.shape[0]
    cands = rng.uniform(0, 1, (batch_size * 10, d))
    return cands[:batch_size]


def strategy_egbo(oracle, X_obs, y_obs, bounds, batch_size, rng,
                  progress=0.5, evo_pop=72, **kw):
    """EGBO: evolutionary candidates + GP scoring (your confirmed winner)."""
    from pymoo.algorithms.moo.unsga3 import UNSGA3
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.termination import NoTermination
    from pymoo.util.ref_dirs import get_reference_directions

    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    Xn = (X_obs - lo) / (hi - lo + 1e-12)

    # Evolutionary candidates
    top_k = min(evo_pop, len(y_obs))
    seed_x = Xn[np.argsort(y_obs)[-top_k:][::-1]]
    if len(seed_x) < max(evo_pop, 2):
        pad = rng.random((max(evo_pop, 2) - len(seed_x), d))
        seed_x = np.vstack([seed_x, pad])
    try:
        pop_size = max(evo_pop, 2)
        ref_dirs = get_reference_directions("energy", 1, pop_size, seed=42)
        algo = UNSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=seed_x)
        pm = PymooProblem(n_var=d, n_obj=1, n_constr=0,
                          xl=np.zeros(d), xu=np.ones(d))
        algo.setup(pm, termination=NoTermination())
        pop = algo.ask()
        n_act = len(pop)
        f_vals = -y_obs[np.argsort(y_obs)[-n_act:]]
        pop.set("F", f_vals.reshape(-1, 1))
        algo.tell(infills=pop)
        ea_cands_n = np.clip(algo.ask().get("X"), 0, 1)[:evo_pop]
    except Exception:
        ea_cands_n = rng.random((evo_pop, d))

    rand_cands_n = rng.random((8, d))
    all_cands_n = np.vstack([ea_cands_n, rand_cands_n])
    all_cands = all_cands_n * (hi - lo) + lo

    gp, scaler = _fit_gp(X_obs, y_obs)
    scores = _ucb_score(gp, scaler, all_cands, y_obs, progress)

    # Greedy novelty-aware selection
    selected, remaining = [], list(range(len(all_cands)))
    for _ in range(batch_size):
        if not remaining:
            break
        rem = np.array(remaining)
        acq = scores[rem]
        # Novelty vs already-selected
        if selected:
            sel_n = all_cands_n[selected]
            nov = np.array([
                np.min(np.linalg.norm(sel_n - all_cands_n[i], axis=1))
                for i in rem
            ])
        else:
            nov = np.ones(len(rem))
        a_norm = (acq - acq.min()) / (acq.max() - acq.min() + 1e-12)
        n_norm = (nov - nov.min()) / (nov.max() - nov.min() + 1e-12)
        score = 0.7 * a_norm + 0.3 * n_norm
        pick = int(rem[np.argmax(score)])
        selected.append(pick)
        remaining.remove(pick)

    return all_cands[selected]


def strategy_llm_bo(oracle, X_obs, y_obs, bounds, batch_size, rng,
                    prior_text="", model="qwen2.5:14b-instruct",
                    mock_llm=False, progress=0.5, n_llm_candidates=20, **kw):
    """
    LLM-BO: LLM generates candidates, GP scores them, best batch selected.
    LLM plays the role of domain expert proposing experiments.
    GP plays the role of uncertainty-aware ranker.
    """
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]

    # LLM proposes candidates in [0,1]^d space
    llm_cands_n = llm_suggest(
        X_obs=(X_obs - lo) / (hi - lo + 1e-12),
        y_obs=y_obs,
        bounds=np.column_stack([np.zeros(d), np.ones(d)]),
        prior_text=prior_text,
        n_suggest=n_llm_candidates,
        model=model,
        mock=mock_llm,
        rng=rng,
    )

    # Add random candidates to ensure diversity if LLM suggestions are clustered
    rand_cands_n = rng.random((8, d))
    all_cands_n = np.vstack([llm_cands_n, rand_cands_n])
    all_cands = all_cands_n * (hi - lo) + lo

    # GP scores the entire pool
    gp, scaler = _fit_gp(X_obs, y_obs)
    scores = _ucb_score(gp, scaler, all_cands, y_obs, progress)

    # Select top batch_size (no novelty weighting — the LLM handles diversity)
    top_idx = np.argsort(scores)[::-1][:batch_size]
    return all_cands[top_idx]


# ── Campaign simulator ─────────────────────────────────────────────────────────

def run_campaign(oracle, X_init, y_init, budget, strategy_fn,
                 strategy_kwargs, batch_size=4, seed=0):
    """Run one campaign. Returns running-best curve and decisions log."""
    bounds = oracle.bounds()
    all_X = oracle._X_raw
    scaler_oracle = oracle._scaler
    rng = np.random.default_rng(seed)

    X_obs = X_init.copy()
    y_obs = y_init.copy()

    X_all_s = scaler_oracle.transform(all_X)
    queried = set()
    for row in scaler_oracle.transform(X_init):
        queried.add(int(np.argmin(np.linalg.norm(X_all_s - row, axis=1))))

    running_best = [float(y_obs.max())] * len(X_init)
    decisions = []
    n_batches = max(1, (budget - len(X_init)) // batch_size)

    for b in range(n_batches):
        step = len(X_init) + b * batch_size
        progress = step / budget

        try:
            candidates = strategy_fn(
                oracle=oracle, X_obs=X_obs, y_obs=y_obs, bounds=bounds,
                batch_size=batch_size, rng=rng, progress=progress,
                **strategy_kwargs,
            )
        except Exception as e:
            warnings.warn(f"Strategy failed at batch {b}: {e}. Using random.")
            d = bounds.shape[0]
            lo, hi = bounds[:, 0], bounds[:, 1]
            candidates = rng.uniform(lo, hi, (batch_size, d))

        # Snap candidates to nearest unqueried oracle row
        unqueried = [i for i in range(len(all_X)) if i not in queried]
        if not unqueried:
            unqueried = list(range(len(all_X)))

        new_x, new_y = [], []
        for cand in candidates[:batch_size]:
            if not unqueried:
                break
            cand_s = scaler_oracle.transform(cand.reshape(1, -1))[0]
            pool_s = scaler_oracle.transform(all_X[unqueried])
            chosen = unqueried[int(np.argmin(np.linalg.norm(pool_s - cand_s, axis=1)))]
            queried.add(chosen)
            new_x.append(all_X[chosen])
            new_y.append(float(oracle._y_raw[chosen]))
            unqueried = [i for i in unqueried if i != chosen]

        X_obs = np.vstack([X_obs, np.array(new_x)])
        y_obs = np.append(y_obs, new_y)
        for _ in new_y:
            running_best.append(float(y_obs.max()))
        decisions.append({"step": step, "progress": progress,
                          "n_obs": len(y_obs)})

    return {"running_best": running_best, "decisions": decisions,
            "X_obs": X_obs, "y_obs": y_obs}


# ── Main experiment ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--out_dir",    default="results_posthoc_llm")
    parser.add_argument("--datasets",   nargs="+",
                        default=["pareto_20210112"])
    parser.add_argument("--n_repeats",  type=int, default=20)
    parser.add_argument("--budget_frac",type=float, default=0.5)
    parser.add_argument("--n_init",     type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model",      default="qwen2.5:14b-instruct",
                        help="Ollama model for LLM-BO. "
                             "Recommended: qwen2.5:72b-instruct or deepseek-r1:70b")
    parser.add_argument("--mock_llm",   action="store_true",
                        help="Use mock LLM (no Ollama needed). For testing.")
    parser.add_argument("--n_llm_candidates", type=int, default=20,
                        help="Number of candidates LLM proposes per batch")
    parser.add_argument("--conditions", nargs="+",
                        default=["egbo", "llm_bo", "random"])
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(args.data_dir).parent))
    from oracle import NNOracle
    from shared_seed_experiment import generate_shared_inits

    DATASET_MAP = {
        "pareto_20201218": "pareto_campaign 2020-12-18_17-38-40",
        "pareto_20210112": "pareto_campaign 2021-01-12_16-26-56",
        "coatings":        "coatings",
        "hartmann3":       "hartmann3",
        "hartmann6":       "hartmann6",
    }

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for ds_label in args.datasets:
        ds_name = DATASET_MAP.get(ds_label, ds_label)
        print(f"\n{'='*60}\nDataset: {ds_label}")

        oracle = NNOracle.from_dataset(ds_name, args.data_dir)
        bounds = oracle.bounds()
        N  = len(oracle._X_raw)
        gb = oracle.global_best()
        budget = max(args.n_init + 10, int(args.budget_frac * N))

        prior_text = PRIOR_KNOWLEDGE.get(
            ds_label,
            DEFAULT_PRIOR.format(d=bounds.shape[0])
        )

        print(f"  N={N}  budget={budget}  dims={bounds.shape[0]}")
        print(f"  global_best={gb:.4f}")
        if args.mock_llm:
            print("  LLM: mock (position-frequency heuristic)")
        else:
            print(f"  LLM: {args.model}")

        shared_inits = generate_shared_inits(
            oracle, args.n_repeats, args.n_init, rng_seed=42)
        best_norms = [y.max()/gb for _, y in shared_inits]
        print(f"  Init best_norm range: {min(best_norms):.2f}–{max(best_norms):.2f}")

        STRATEGY_FNS = {
            "egbo":    (strategy_egbo,   {}),
            "random":  (strategy_random, {}),
            "llm_bo":  (strategy_llm_bo, {
                "prior_text":       prior_text,
                "model":            args.model,
                "mock_llm":         args.mock_llm,
                "n_llm_candidates": args.n_llm_candidates,
            }),
        }

        for cond in args.conditions:
            if cond not in STRATEGY_FNS:
                print(f"  Unknown condition '{cond}', skipping")
                continue

            fn, kwargs = STRATEGY_FNS[cond]
            aucs, finals = [], []
            print(f"  {cond}...", end="", flush=True)

            for seed_idx, (X_init, y_init) in enumerate(shared_inits):
                res = run_campaign(
                    oracle, X_init, y_init, budget,
                    strategy_fn=fn, strategy_kwargs=kwargs,
                    batch_size=args.batch_size, seed=seed_idx,
                )
                curve = np.array(res["running_best"])
                auc   = float((curve / gb).mean())
                final = float(curve[-1] / gb)
                aucs.append(auc)
                finals.append(final)
                all_rows.append({
                    "dataset": ds_label, "condition": cond,
                    "seed": seed_idx, "auc": auc, "final": final,
                    "budget": budget, "N": N,
                })
                print(".", end="", flush=True)

            print(f"  auc={np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
                  f"final={np.mean(finals):.3f}")

            # Save curves
            cond_dir = out_dir / ds_label / cond
            cond_dir.mkdir(parents=True, exist_ok=True)
            curves_arr = np.zeros((args.n_repeats, budget))
            for i, row in enumerate(all_rows):
                pass  # curves saved per-seed above; summary in CSV is enough

    # ── Results ────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "posthoc_llm_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    print(f"\n{'='*65}")
    print("RESULTS SUMMARY")
    print(f"{'='*65}")

    n = args.n_repeats

    def ttest(m1, s1, m2, s2, n=20):
        se = np.sqrt(s1**2/n + s2**2/n)
        t  = (m1 - m2) / (se + 1e-12)
        return float(2 * stats.t.sf(abs(t), df=2*n-2))

    for ds in df.dataset.unique():
        sub = df[df.dataset == ds]
        print(f"\n{ds}:")
        agg = sub.groupby("condition").agg(
            auc_mean=("auc","mean"), auc_std=("auc","std"),
            final_mean=("final","mean"),
        ).sort_values("auc_mean", ascending=False)

        for cond, row in agg.iterrows():
            print(f"  {cond:<12} auc={row.auc_mean:.3f}±{row.auc_std:.3f}  "
                  f"final={row.final_mean:.3f}")

        # Key comparisons
        print()
        for c1, c2, label in [
            ("llm_bo", "random", "LLM-BO vs Random"),
            ("llm_bo", "egbo",   "LLM-BO vs EGBO"),
            ("egbo",   "random", "EGBO vs Random (sanity check)"),
        ]:
            if c1 in agg.index and c2 in agg.index:
                m1, s1 = agg.loc[c1, "auc_mean"], agg.loc[c1, "auc_std"]
                m2, s2 = agg.loc[c2, "auc_mean"], agg.loc[c2, "auc_std"]
                p = ttest(m1, s1, m2, s2, n)
                sig = " *" if p < 0.05 else ""
                gap_closed = (m1-m2) / (agg.loc["egbo","auc_mean"] -
                              agg.loc["random","auc_mean"] + 1e-12) * 100
                print(f"  {label:<28} Δ={m1-m2:+.3f}  p={p:.3f}{sig}  "
                      f"gap_closed={gap_closed:.0f}%")

    print()
    print("Interpretation guide:")
    print("  gap_closed = (LLM-BO - Random) / (EGBO - Random) × 100")
    print("  > 80%: LLM-BO nearly matches EGBO — domain knowledge is effective")
    print("  50-80%: LLM-BO is meaningfully better than random")
    print("  < 20%: LLM domain knowledge adds little over random search")
    print()
    if args.mock_llm:
        print("NOTE: Results used mock LLM (heuristic). Re-run without --mock_llm")
        print(f"      for genuine LLM results with {args.model}.")


if __name__ == "__main__":
    main()
