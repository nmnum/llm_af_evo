"""
oracle_gap_experiment.py — (a) default vs (b) LLM-tuned vs (c) oracle-tuned
Bayesian-GP EGBO, isolating exactly 3 LLM judgment calls:
    ls prior mode, ARD flag, evo population size.

Everything else is held fixed across all three conditions:
    ls prior concentration   = 5.0
    noise prior mode         = 0.188 (median true noise across datasets)
    noise prior concentration= 2.0
    acquisition transitions  = fixed 30% / 60% schedule (same in a/b/c)

Run at both budget_frac=0.5 and budget_frac=1.0 — report separately, not pooled.

Primary diagnostic per seed:
    ls_mode_error    = |log(oracle_ls_mode) - log(chosen_ls_mode)|
    ard_match        = chosen_ard_flag == oracle_ard_flag
    evo_pop_error    = |oracle_evo_pop - chosen_evo_pop| / oracle_evo_pop

Usage:
    python oracle_gap_experiment.py --data_dir data/ --out_dir results_oracle_gap/ \
        --datasets pareto_20210112 hartmann6 coatings \
        --budget_fracs 0.5 1.0 --n_repeats 20 --model qwen2.5:14b-instruct
"""

import argparse
import json
import pathlib
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Fixed, never exposed to any condition
LS_PRIOR_CONC      = 5.0
NOISE_PRIOR_MODE   = 0.188   # median true noise across formulation/pareto/coatings
NOISE_PRIOR_CONC   = 2.0
DEFAULT_LS_MODE    = 1.0     # geo-mean of true ls_norm across datasets
ACQF_SWITCH_EARLY  = 0.30    # LHS/random -> qNEHVI-equivalent (EGBO acq)
ACQF_SWITCH_LATE   = 0.60    # qNEHVI-equivalent -> exploitation (low-beta UCB/EI)
DEFAULT_EVO_POP    = 300


def gamma_rate_from_mode(mode, concentration):
    """Gamma mode = (concentration-1)/rate  =>  rate = (concentration-1)/mode."""
    mode = max(float(mode), 1e-6)
    return (concentration - 1.0) / mode


# ── True landscape parameters (oracle source) ─────────────────────────────────

def fit_true_landscape_params(oracle, n_restarts=5):
    """
    Fit a GP on the FULL dataset (all N points) to extract ground-truth
    lengthscale, noise, and ARD ratio. This is the oracle's knowledge source.
    Uses sklearn with WhiteKernel for an explicit noise estimate.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    X = oracle._X_raw
    y = oracle._y_raw
    bounds = oracle.bounds()
    d = bounds.shape[0]

    lo, hi = bounds[:, 0], bounds[:, 1]
    Xn = (X - lo) / (hi - lo + 1e-12)
    y_s = (y - y.mean()) / (y.std() + 1e-12)

    # Isotropic fit -> true ls_norm, true noise
    kernel_iso = Matern(nu=2.5, length_scale=1.0,
                        length_scale_bounds=(1e-3, 1e3)) \
                 + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_iso = GaussianProcessRegressor(kernel=kernel_iso, normalize_y=False,
                                          n_restarts_optimizer=n_restarts)
        gp_iso.fit(Xn, y_s)

    ls_true    = float(gp_iso.kernel_.k1.length_scale)
    noise_true = float(gp_iso.kernel_.k2.noise_level)

    # ARD fit -> per-dimension lengthscales, to decide true ARD flag
    kernel_ard = Matern(nu=2.5, length_scale=np.ones(d),
                        length_scale_bounds=(1e-3, 1e3)) \
                 + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_ard = GaussianProcessRegressor(kernel=kernel_ard, normalize_y=False,
                                          n_restarts_optimizer=n_restarts)
        gp_ard.fit(Xn, y_s)
    ls_per_dim = np.atleast_1d(gp_ard.kernel_.k1.length_scale)
    ard_ratio  = float(ls_per_dim.max() / (ls_per_dim.min() + 1e-12))
    ard_true   = ard_ratio > 2.0

    # Evo population oracle rule (from pool-composition ablation: ls_norm threshold)
    evo_pop_true = 300 if ls_true > 0.15 else 50

    return {
        "ls_true": ls_true, "noise_true": noise_true,
        "ard_true": ard_true, "ard_ratio": ard_ratio,
        "evo_pop_true": evo_pop_true,
    }


# ── Bayesian GP (GPyTorch) ─────────────────────────────────────────────────────

def fit_bayesian_gp(X_obs, y_obs, bounds, ls_mode, ard, evo_pop_unused=None):
    """Fit GPyTorch ExactGP with fixed-concentration Gamma priors, given mode."""
    import torch
    import gpytorch
    from gpytorch.kernels import ScaleKernel, MaternKernel
    from gpytorch.means import ConstantMean
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.priors import GammaPrior
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.models.gpytorch import GPyTorchModel
    # fit_gpytorch_mll is the stable top-level entry point across botorch
    # versions; it dispatches to scipy L-BFGS-B internally and does not take
    # max_attempts (that kwarg belongs to a different, version-specific
    # internal helper that does not exist in this installed version).
    from botorch.fit import fit_gpytorch_mll

    n, d = X_obs.shape
    lo, hi = bounds[:, 0], bounds[:, 1]
    Xn = (X_obs - lo) / (hi - lo + 1e-12)
    y_mean, y_std = y_obs.mean(), y_obs.std() + 1e-12
    y_s = (y_obs - y_mean) / y_std

    train_X = torch.tensor(Xn, dtype=torch.float64)
    train_y = torch.tensor(y_s, dtype=torch.float64)

    ls_rate    = gamma_rate_from_mode(ls_mode, LS_PRIOR_CONC)
    noise_rate = gamma_rate_from_mode(NOISE_PRIOR_MODE, NOISE_PRIOR_CONC)

    likelihood = GaussianLikelihood(
        noise_prior=GammaPrior(concentration=NOISE_PRIOR_CONC, rate=noise_rate)
    )

    class _GP(GPyTorchModel, gpytorch.models.ExactGP):
        _num_outputs = 1  # required by the GPyTorchModel mixin

        def __init__(self):
            super().__init__(train_X, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=d if ard else None,
                    lengthscale_prior=GammaPrior(concentration=LS_PRIOR_CONC,
                                                  rate=ls_rate),
                )
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x))

    model = _GP()

    # CRITICAL FIX: GammaPrior registers a prior for MAP optimisation, it does
    # NOT set the parameter's starting value. Without this explicit init, the
    # kernel keeps GPyTorch's framework default (~1.0) regardless of ls_mode,
    # so "default" and "oracle" conditions can silently converge to the same
    # place if optimisation also fails. Explicitly initialise at the mode.
    model.covar_module.base_kernel.initialize(lengthscale=float(ls_mode))
    model.likelihood.initialize(noise=float(NOISE_PRIOR_MODE))

    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_failed = False
    fit_error_msg = None
    try:
        fit_gpytorch_mll(mll)
    except Exception as e:
        fit_failed = True
        fit_error_msg = f"{type(e).__name__}: {e}"
        import warnings as _w
        _w.warn(f"GP fit failed: {fit_error_msg}")

    return model, likelihood, lo, hi, y_mean, y_std, fit_failed, fit_error_msg


def predict_bayesian_gp(model, likelihood, X_cand, bounds, y_mean, y_std):
    import torch
    lo, hi = bounds[:, 0], bounds[:, 1]
    Xn = (X_cand - lo) / (hi - lo + 1e-12)
    test_X = torch.tensor(Xn, dtype=torch.float64)
    model.eval(); likelihood.eval()
    with torch.no_grad():
        post = likelihood(model(test_X))
        mu = post.mean.numpy() * y_std + y_mean
        sigma = post.variance.sqrt().numpy() * y_std
    return mu, sigma


# ── EGBO-style campaign with controllable knobs ───────────────────────────────

def run_egbo_campaign(oracle, X_init, y_init, budget, ls_mode_fn, ard_fn,
                      evo_pop_fn, random_state=0, batch_size=4):
    """
    ls_mode_fn(X_obs, y_obs, step)   -> ls prior mode to use this batch
    ard_fn(X_obs, y_obs, step)       -> bool, ARD on/off this batch
    evo_pop_fn(X_obs, y_obs, step)   -> int, evolutionary population size
    Acquisition schedule fixed at ACQF_SWITCH_EARLY / ACQF_SWITCH_LATE (all conditions).
    """
    from pymoo.algorithms.moo.unsga3 import UNSGA3
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.termination import NoTermination
    from pymoo.util.ref_dirs import get_reference_directions

    warnings.filterwarnings("ignore")
    bounds = oracle.bounds()
    d = bounds.shape[0]
    N_dataset = len(oracle._X_raw)
    all_X = oracle._X_raw
    scaler_oracle = oracle._scaler
    lo, hi = bounds[:, 0], bounds[:, 1]

    X_obs, y_obs = X_init.copy(), y_init.copy()
    queried = set()
    X_all_s = scaler_oracle.transform(all_X)
    for row in scaler_oracle.transform(X_init):
        queried.add(int(np.argmin(np.linalg.norm(X_all_s - row, axis=1))))

    running_best = [float(y_obs.max())] * len(X_init)
    decisions = []
    n_batches = max(1, (budget - len(X_init)) // batch_size)

    rng = np.random.default_rng(random_state)

    for b in range(n_batches):
        step = len(X_init) + b * batch_size
        progress = step / budget

        ls_mode  = ls_mode_fn(X_obs, y_obs, step)
        ard      = ard_fn(X_obs, y_obs, step)
        evo_pop  = evo_pop_fn(X_obs, y_obs, step)

        model, likelihood, _, _, y_mean, y_std, fit_failed, fit_err = fit_bayesian_gp(
            X_obs, y_obs, bounds, ls_mode, ard)
        if fit_failed and b == 0:  # print full traceback once per campaign, first batch only
            import traceback as _tb
            print(f"    [DEBUG] first GP fit failure: {fit_err}")

        # Candidate pool: evolutionary + random, size = evo_pop (+10 random)
        Xn_obs = (X_obs - lo) / (hi - lo + 1e-12)
        top_k = min(evo_pop, len(y_obs))
        top_idx = np.argsort(y_obs)[-top_k:][::-1]
        seed_x = Xn_obs[top_idx]
        if seed_x.shape[0] < max(evo_pop, 2):
            pad = rng.random((max(evo_pop, 2) - seed_x.shape[0], d))
            seed_x = np.vstack([seed_x, pad])
        try:
            pop_size = max(evo_pop, 2)
            ref_dirs = get_reference_directions("energy", 1, pop_size,
                                                seed=random_state + b)
            algo = UNSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=seed_x)
            pm = PymooProblem(n_var=d, n_obj=1, n_constr=0,
                              xl=np.zeros(d), xu=np.ones(d))
            algo.setup(pm, termination=NoTermination())
            pop = algo.ask()
            n_act = len(pop)
            f_vals = -y_obs[np.argsort(y_obs)[-n_act:]]
            pop.set("F", f_vals.reshape(-1, 1))
            algo.tell(infills=pop)
            ea_cands = np.clip(algo.ask().get("X"), 0, 1)[:evo_pop]
        except Exception:
            ea_cands = rng.random((evo_pop, d))

        rand_cands = rng.random((10, d))
        cands_n = np.vstack([ea_cands, rand_cands])
        cands_raw = cands_n * (hi - lo) + lo

        mu, sigma = predict_bayesian_gp(model, likelihood, cands_raw, bounds,
                                        y_mean, y_std)

        # Fixed acquisition schedule, identical across a/b/c
        if progress < ACQF_SWITCH_EARLY:
            beta = 5.0       # broad explore
        elif progress < ACQF_SWITCH_LATE:
            beta = 1.0       # EGBO-equivalent balance
        else:
            beta = 0.1       # exploit
        scores = mu + beta * sigma

        order = np.argsort(scores)[::-1]
        unqueried = [i for i in range(N_dataset) if i not in queried]
        if not unqueried:
            unqueried = list(range(N_dataset))

        picked, new_x, new_y = 0, [], []
        for idx in order:
            if picked >= batch_size:
                break
            x_raw = cands_raw[idx]
            x_s = scaler_oracle.transform(x_raw.reshape(1, -1))[0]
            pool_s = scaler_oracle.transform(all_X[unqueried])
            chosen = unqueried[int(np.argmin(np.linalg.norm(pool_s - x_s, axis=1)))]
            queried.add(chosen)
            new_x.append(all_X[chosen])
            new_y.append(float(oracle._y_raw[chosen]))
            unqueried = [i for i in unqueried if i != chosen]
            picked += 1

        X_obs = np.vstack([X_obs, np.array(new_x)])
        y_obs = np.append(y_obs, new_y)
        for _ in new_y:
            running_best.append(float(y_obs.max()))

        decisions.append({
            "step": step, "ls_mode": ls_mode, "ard": ard,
            "evo_pop": evo_pop, "beta": beta, "fit_failed": fit_failed,
        })

    return {"running_best": running_best, "decisions": decisions, "X_obs": X_obs,
            "y_obs": y_obs}


# ── Condition definitions ──────────────────────────────────────────────────────

def make_default_fns(true_params, n_dims):
    def ls_mode_fn(X, y, step):
        return DEFAULT_LS_MODE
    def ard_fn(X, y, step):
        return len(y) >= 3 * n_dims
    def evo_pop_fn(X, y, step):
        return DEFAULT_EVO_POP
    return ls_mode_fn, ard_fn, evo_pop_fn


def make_oracle_fns(true_params, n_dims):
    def ls_mode_fn(X, y, step):
        return true_params["ls_true"]
    def ard_fn(X, y, step):
        return true_params["ard_true"]
    def evo_pop_fn(X, y, step):
        return true_params["evo_pop_true"]
    return ls_mode_fn, ard_fn, evo_pop_fn


def make_llm_fns(model, n_dims, log_list):
    """LLM chooses ls_mode, ard, evo_pop from partial campaign observations."""
    import ollama, re, json as _json

    SYSTEM = """You tune a Bayesian GP for Bayesian optimisation, given partial
campaign observations. Respond with JSON only, no markdown, no explanation:
{"ls_mode": <float 0.05-5.0>, "ard": <true/false>, "evo_pop": <int 30-400>}

Guidance:
  ls_mode: GP lengthscale prior center. Small (0.1-0.5) = rough/noisy landscape.
    Large (1.5-3.0) = smooth landscape. Estimate from how much y varies between
    nearby x values in the observations.
  ard: true if different input dimensions seem to matter very differently
    (some dims barely affect y, others strongly affect it). false if dims
    seem roughly equally important. Only consider true if n_obs >= 3*n_dims.
  evo_pop: evolutionary candidate pool size. Use ~300 if landscape looks
    structured/multimodal (y varies non-monotonically). Use ~50 if landscape
    looks flat or noisy (y looks unrelated to x)."""

    def _decide(X, y, step):
        n = len(y)
        # Build compact summary of observations
        y_sorted = np.sort(y)[::-1][:5]
        prompt = (f"n_obs={n}, n_dims={n_dims}, step={step}\n"
                 f"y range: [{y.min():.3f}, {y.max():.3f}], "
                 f"y std: {y.std():.3f}\n"
                 f"top-5 y values: {y_sorted.tolist()}\n"
                 f"Choose ls_mode, ard, evo_pop as JSON.")
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 128, "think": False},
            )
            text = resp["message"]["content"]
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            text = text.strip().strip("`").replace("json\n", "").strip()
            parsed = _json.loads(text)
            ls_mode = float(np.clip(parsed.get("ls_mode", DEFAULT_LS_MODE),
                                    0.05, 5.0))
            ard = bool(parsed.get("ard", n >= 3 * n_dims))
            evo_pop = int(np.clip(parsed.get("evo_pop", DEFAULT_EVO_POP), 30, 400))
        except Exception:
            ls_mode, ard, evo_pop = DEFAULT_LS_MODE, n >= 3 * n_dims, DEFAULT_EVO_POP

        log_list.append({"step": step, "ls_mode": ls_mode, "ard": ard,
                         "evo_pop": evo_pop})
        return ls_mode, ard, evo_pop

    cache = {}
    def ls_mode_fn(X, y, step):
        if step not in cache:
            cache[step] = _decide(X, y, step)
        return cache[step][0]
    def ard_fn(X, y, step):
        if step not in cache:
            cache[step] = _decide(X, y, step)
        return cache[step][1]
    def evo_pop_fn(X, y, step):
        if step not in cache:
            cache[step] = _decide(X, y, step)
        return cache[step][2]

    return ls_mode_fn, ard_fn, evo_pop_fn


# ── Main experiment loop ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="results_oracle_gap")
    parser.add_argument("--datasets", nargs="+",
                        default=["pareto_20210112", "hartmann6", "coatings"])
    parser.add_argument("--budget_fracs", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--n_repeats", type=int, default=20)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--model", default="qwen2.5:14b-instruct")
    parser.add_argument("--llm_call_interval", type=int, default=10,
                        help="LLM re-decides every N steps (not every batch)")
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from oracle import NNOracle
    from shared_seed_experiment import generate_shared_inits

    DATASET_MAP = {
        "coatings":        "coatings",
        "pareto_20201218": "pareto_campaign 2020-12-18_17-38-40",
        "pareto_20210112": "pareto_campaign 2021-01-12_16-26-56",
        "hartmann3":       "hartmann3",
        "hartmann6":       "hartmann6",
    }

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for ds_label in args.datasets:
        ds_name = DATASET_MAP[ds_label]
        oracle = NNOracle.from_dataset(ds_name, args.data_dir)
        bounds = oracle.bounds()
        n_dims = bounds.shape[0]
        N = len(oracle._X_raw)
        gb = oracle.global_best()

        print(f"\n{'='*60}\n{ds_label}: fitting true landscape params (oracle source)...")
        true_params = fit_true_landscape_params(oracle)
        print(f"  ls_true={true_params['ls_true']:.3f}  "
              f"noise_true={true_params['noise_true']:.3f}  "
              f"ard_true={true_params['ard_true']} "
              f"(ratio={true_params['ard_ratio']:.2f})  "
              f"evo_pop_true={true_params['evo_pop_true']}")

        for budget_frac in args.budget_fracs:
            budget = max(args.n_init + 10, int(budget_frac * N))
            shared_inits = generate_shared_inits(oracle, args.n_repeats,
                                                  args.n_init, rng_seed=42)

            for cond_name, fn_maker in [
                ("default", make_default_fns),
                ("oracle",  make_oracle_fns),
                ("llm",     None),  # special-cased below
            ]:
                print(f"  budget_frac={budget_frac} condition={cond_name} ...")
                llm_log = []
                if cond_name == "llm":
                    ls_fn, ard_fn, evo_fn = make_llm_fns(args.model, n_dims, llm_log)
                else:
                    ls_fn, ard_fn, evo_fn = fn_maker(true_params, n_dims)

                for seed_idx, (X_init, y_init) in enumerate(shared_inits):
                    try:
                        res = run_egbo_campaign(
                            oracle, X_init, y_init, budget,
                            ls_fn, ard_fn, evo_fn,
                            random_state=seed_idx,
                        )
                        curve = np.array(res["running_best"])
                        auc = float((curve / gb).mean())
                        final = float(curve[-1] / gb)

                        # Diagnostics vs oracle ground truth
                        last_dec = res["decisions"][-1] if res["decisions"] else {}
                        chosen_ls = last_dec.get("ls_mode", DEFAULT_LS_MODE)
                        chosen_ard = last_dec.get("ard", False)
                        chosen_evo = last_dec.get("evo_pop", DEFAULT_EVO_POP)

                        ls_mode_error = abs(
                            np.log(max(true_params["ls_true"], 1e-6)) -
                            np.log(max(chosen_ls, 1e-6))
                        )
                        ard_match = (chosen_ard == true_params["ard_true"])
                        evo_pop_error = abs(
                            true_params["evo_pop_true"] - chosen_evo
                        ) / true_params["evo_pop_true"]

                        n_fit_failed = sum(
                            1 for dec in res["decisions"] if dec.get("fit_failed"))
                        rows.append({
                            "dataset": ds_label, "budget_frac": budget_frac,
                            "condition": cond_name, "seed": seed_idx,
                            "auc": auc, "final": final,
                            "ls_mode_error": ls_mode_error,
                            "ard_match": ard_match,
                            "evo_pop_error": evo_pop_error,
                            "chosen_ls": chosen_ls, "true_ls": true_params["ls_true"],
                            "n_fit_failed": n_fit_failed,
                            "n_batches": len(res["decisions"]),
                        })
                    except Exception as e:
                        print(f"    seed {seed_idx} failed: {e}")

                if cond_name == "llm" and llm_log:
                    with open(out_dir / f"{ds_label}_bf{budget_frac}_llm_log.json",
                             "w") as f:
                        json.dump(llm_log, f, indent=2)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "oracle_gap_summary.csv", index=False)
    print(f"\nSaved: {out_dir}/oracle_gap_summary.csv")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}\nRESULTS SUMMARY\n{'='*70}")
    for ds in df.dataset.unique():
        for bf in df.budget_frac.unique():
            sub = df[(df.dataset == ds) & (df.budget_frac == bf)]
            if sub.empty:
                continue
            print(f"\n{ds}  budget_frac={bf}:")
            agg = sub.groupby("condition").agg(
                auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                ls_err_mean=("ls_mode_error", "mean"),
                ard_match_rate=("ard_match", "mean"),
                fit_fail_total=("n_fit_failed", "sum"),
                batches_total=("n_batches", "sum"),
            )
            for cond in ["default", "llm", "oracle"]:
                if cond not in agg.index:
                    continue
                r = agg.loc[cond]
                fail_pct = (100 * r.fit_fail_total / r.batches_total
                           if r.batches_total else 0)
                warn = "  *** GP FIT FAILING ***" if fail_pct > 5 else ""
                print(f"  {cond:<10} auc={r.auc_mean:.3f}±{r.auc_std:.3f}  "
                      f"ls_mode_err={r.ls_err_mean:.2f}  "
                      f"ard_match={r.ard_match_rate:.0%}  "
                      f"fit_fail={fail_pct:.0f}%{warn}")

            if all(c in agg.index for c in ["default", "llm", "oracle"]):
                gap_total = agg.loc["oracle", "auc_mean"] - agg.loc["default", "auc_mean"]
                gap_closed = agg.loc["llm", "auc_mean"] - agg.loc["default", "auc_mean"]
                frac = gap_closed / gap_total if abs(gap_total) > 1e-6 else float('nan')
                print(f"  → LLM closes {frac:.0%} of the default-to-oracle gap "
                      f"({gap_closed:+.3f} / {gap_total:+.3f})")


if __name__ == "__main__":
    main()
