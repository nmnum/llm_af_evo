"""
strategy_llm_candidate_gen.py — LLM-as-candidate-generator strategy for comparison.

NOTE: this was originally named/labeled "LABO" (strategy_llm_labo.py,
mo_llm_labo), but it does NOT implement the actual LABO framework from the
literature — real LABO uses a multi-fidelity Kennedy-O'Hagan surrogate
(f_R(x) = rho*f_L(x) + delta(x)) with a gating criterion that can SKIP a real
experiment entirely when the LLM's low-fidelity prediction is trusted enough
(p_delta(x*) <= tau). This strategy always runs the full batch_size of real
experiments every batch — it never skips one — so it's an LLM candidate
generator merged into EGBO's acquisition loop, not a gated multi-fidelity
method. Renamed to avoid the false claim of literature fidelity. A true
KOH-gated LABO implementation is a separate, larger scope item, not yet built.

Implements the user's original proposal: every batch, the LLM generates
20-30 candidate formulations, these are merged with EGBO's evolutionary
candidates, and qLogNEHVI scores the entire pool to select the final batch.

This is the "LLM as candidate generator throughout the campaign" architecture,
contrasted with LS-NA-EGBO's "LLM as one-shot warm-start" architecture.

Key difference from the existing strategy_mo_llm in excipient_campaign_mo.py:
  - Uses qLogNEHVI (BoTorch) for scoring instead of the trust-weighted mixing
  - Uses novelty-aware selection for the final batch pick
  - No trust diagnostic (which was shown to be unreliable at small N)
  - LLM candidates are scored on the same acquisition function as EGBO candidates

Usage:
    from strategy_llm_candidate_gen import strategy_mo_llm_candidate_gen

    # In STRATEGIES dict:
    "mo_llm_candidate_gen": (strategy_mo_llm_candidate_gen,
                              {"prior_text": ..., "model": ..., "mock_llm": True}),
"""

import json
import re
import warnings
import numpy as np
import torch

from excipient_campaign_mo import (
    pareto_front_of, OBJECTIVE_DIRECTIONS, OBJECTIVE_NAMES,
    strategy_mo_egbo,
)
from excipient_oracle import (
    AMINO_ACIDS, SUGARS, SURFACTANTS,
    AA_LIST, SUGAR_LIST, SF_LIST,
    formulation_to_vector, vector_to_formulation,
)
from novelty_selection import novelty_aware_select_vectorised


def _snap(val, props):
    """Snap a concentration value to the valid step grid."""
    val = float(val)
    val = round(val / props["step"]) * props["step"]
    return float(np.clip(val, props["min"], props["max"]))


def _best_match(s, opts):
    """Fuzzy-match an excipient name."""
    s = str(s).lower().replace(" ", "").replace("-", "")
    for o in opts:
        o_clean = o.replace(" ", "").replace("-", "")
        if o_clean in s or s in o_clean:
            return o
    return opts[0]


_TARGET_CODE_TO_NAME = {"T": "Tm", "K": "kD", "V": "viscosity"}


def _decode_targets(raw_targets):
    """Decode single-letter target codes (T/K/V) back to full objective
    names, matching llm_warmstart.py's schema fix. Passes through
    unrecognised values as-is in case the model ignored the code
    instruction and used full names."""
    if not isinstance(raw_targets, list):
        return []
    return [_TARGET_CODE_TO_NAME.get(str(t).strip().upper(), str(t)) for t in raw_targets]


def _llm_generate_candidates(
    X_obs, Y_obs, prior_text, n_propose, model, mock, rng,
    max_retries=3,
):
    """
    Ask the LLM to propose n_propose formulations given current observations.
    Returns list of formulation dicts.
    """
    from excipient_campaign_mo import _fmt_formulation

    # Show current Pareto front
    if len(Y_obs) > 0:
        Y = Y_obs.copy()
        for j, d in enumerate(OBJECTIVE_DIRECTIONS):
            if d == "min":
                Y[:, j] = -Y[:, j]
        ge = (Y[None, :, :] >= Y[:, None, :]).all(axis=2)
        gt = (Y[None, :, :] > Y[:, None, :]).any(axis=2)
        dominated = (ge & gt).any(axis=1)
        pareto_idx = np.where(~dominated)[0]
    else:
        pareto_idx = np.array([], dtype=int)

    # Show more of the Pareto front than before (was capped at 8, arbitrary
    # index order). Sort by first-objective value so the shown subset spans
    # the front rather than clustering wherever the index order happens to
    # put them, and raise the cap now that num_predict has more headroom.
    obs_lines = []
    if len(pareto_idx) > 0:
        pf_sort_order = pareto_idx[np.argsort(Y_obs[pareto_idx, 0])]
    else:
        pf_sort_order = pareto_idx
    n_shown = min(15, len(pf_sort_order))
    shown_idx = pf_sort_order[np.linspace(0, len(pf_sort_order) - 1, n_shown, dtype=int)] \
        if n_shown > 0 else pf_sort_order
    for i in shown_idx:
        form = vector_to_formulation(X_obs[i])
        obs_lines.append(f"  {_fmt_formulation(form, Y_obs[i])}  [Pareto]")

    aa_opts = ", ".join(f"{a}({AMINO_ACIDS[a]['min']}-{AMINO_ACIDS[a]['max']}mM)"
                        for a in AA_LIST)
    sug_opts = ", ".join(f"{s}({SUGARS[s]['min']}-{SUGARS[s]['max']}mM)"
                         for s in SUGAR_LIST)
    sf_opts = ", ".join(f"{sf}({SURFACTANTS[sf]['min']}-{SURFACTANTS[sf]['max']}%)"
                        for sf in SF_LIST)

    prompt = f"""{prior_text.strip()}

Current Pareto front ({len(pareto_idx)} non-dominated formulations so far,
{n_shown} shown below spanning the front):
{chr(10).join(obs_lines) if obs_lines else "  (none yet — this is the first batch)"}

Available excipients:
  Amino acids (choose one): {aa_opts}
  Sugars (choose one): {sug_opts}
  Surfactants (choose one): {sf_opts}
  EDTA: true or false

Propose exactly {n_propose} new formulations to test next. For each, state
which objective(s) it primarily targets using the single-letter codes T (Tm),
K (kD), V (viscosity) — e.g. ["T","K"] for a formulation targeting both Tm
and kD. State any trade-off it makes in AT MOST 10 WORDS TOTAL — never
exceed 10 words regardless of how many objectives are involved.

Respond with JSON only:
{{
  "candidates": [
    {{"aa": "<name>", "aa_conc": <mM>, "sugar": "<name>", "sugar_conc": <mM>,
      "surfactant": "<name>", "surfactant_conc": <pct>, "edta": <true/false>,
      "targets": ["T"|"K"|"V", ...],
      "tradeoff": "<max 10 words total>"}}
  ]
}}"""

    if mock:
        # Mock: perturb Pareto front points + random exploration
        proposals = []
        for _ in range(n_propose):
            if len(pareto_idx) > 0 and rng.random() < 0.6:
                base_i = pareto_idx[rng.integers(len(pareto_idx))]
                base = vector_to_formulation(X_obs[base_i])
            else:
                aa = str(rng.choice(AA_LIST))
                base = {"aa": aa, "aa_conc": AMINO_ACIDS[aa]["optimal_conc"],
                       "sugar": str(rng.choice(SUGAR_LIST)),
                       "sugar_conc": 50.0,
                       "surfactant": str(rng.choice(SF_LIST)),
                       "surfactant_conc": 0.4, "edta": bool(rng.random() > 0.5)}
            aa_p = AMINO_ACIDS[base["aa"]]
            base = dict(base)
            base["aa_conc"] = _snap(base["aa_conc"] + rng.normal(0, aa_p["breadth"]*0.3),
                                   aa_p)
            base.setdefault("targets", [])
            base.setdefault("tradeoff", "")
            proposals.append(base)
        return proposals, {"llm_fallback_used": False, "llm_retry_count": 0}

    import ollama
    text = ""
    retry_count = 0
    for attempt in range(max_retries):
        retry_count = attempt
        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a pharmaceutical formulation scientist. "
                                "Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.3, "num_predict": 4096, "think": False,
                          "seed": int(rng.integers(1_000_000)) + attempt},
            )
            text = resp["message"]["content"]
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            text = re.sub(r'```(?:json)?', '', text).strip().strip('`')
            text = re.sub(r',(\s*[}\]])', r'\1', text)
            parsed = json.loads(text)
            raw = parsed.get("candidates", [])

            proposals = []
            for c in raw:
                try:
                    aa = _best_match(c.get("aa", ""), AA_LIST)
                    sug = _best_match(c.get("sugar", ""), SUGAR_LIST)
                    sf = _best_match(c.get("surfactant", ""), SF_LIST)
                    form = {
                        "aa": aa,
                        "aa_conc": _snap(c.get("aa_conc", AMINO_ACIDS[aa]["optimal_conc"]),
                                       AMINO_ACIDS[aa]),
                        "sugar": sug,
                        "sugar_conc": _snap(c.get("sugar_conc", SUGARS[sug]["optimal_conc"]),
                                          SUGARS[sug]),
                        "surfactant": sf,
                        "surfactant_conc": _snap(c.get("surfactant_conc",
                                                      SURFACTANTS[sf]["optimal_conc"]),
                                                SURFACTANTS[sf]),
                        "edta": bool(c.get("edta", False)),
                        "targets": _decode_targets(c.get("targets", [])),
                        "tradeoff": " ".join(str(c.get("tradeoff", "")).split()[:10]),
                    }
                    proposals.append(form)
                except Exception:
                    continue

            if proposals:
                return proposals[:n_propose], {"llm_fallback_used": False,
                                                "llm_retry_count": retry_count}

        except Exception as e:
            if attempt == max_retries - 1:
                warnings.warn(f"LLM candidate-gen proposal failed: {e}")

    # Fallback: mock
    fallback_proposals, _ = _llm_generate_candidates(
        X_obs, Y_obs, prior_text, n_propose, model, True, rng)
    return fallback_proposals, {"llm_fallback_used": True, "llm_retry_count": retry_count}


def strategy_mo_llm_candidate_gen(oracle, X_obs, Y_obs, bounds, batch_size, rng,
                          prior_text="", model="qwen3:32b",
                          mock_llm=False, n_llm_candidates=20,
                          w_acq=0.9, w_nov=0.1,
                          n_init=None, budget=None, llm_taper_frac=0.5,
                          **kw):
    """
    LLM-as-candidate-generator: LLM generates candidates every batch, merged
    with EGBO evolutionary candidates, scored by qLogNEHVI, selected with
    novelty-aware selection.

    This is the "LLM throughout the campaign" architecture for comparison
    against LS-NA-EGBO's "LLM warm-start only" architecture. NOTE: this does
    not implement the literature LABO gating mechanism — see the module
    docstring.

    Tapering: per Cisse et al., LLM/BO hybrid advantage concentrates in the
    early campaign and BO catches up later, so calling the LLM every batch
    for the whole campaign pays full LLM cost for batches where it likely
    adds little. If both n_init and budget are provided, the LLM is only
    called for the first `llm_taper_frac` fraction of batches; later batches
    fall back to EGBO-only (LLM candidate pool empty for that batch). If
    n_init/budget aren't provided, tapering is disabled (always calls the
    LLM), preserving the original every-batch behaviour.
    """
    lo, hi = bounds[:, 0], bounds[:, 1]
    d = bounds.shape[0]

    # ── Tapering: decide whether this batch calls the LLM at all ──
    call_llm_this_batch = True
    if n_init is not None and budget is not None and batch_size > 0:
        n_batches_total = max(1, (budget - n_init) // batch_size)
        batch_idx = max(0, (len(X_obs) - n_init) // batch_size)
        taper_cutoff = max(1, int(round(n_batches_total * llm_taper_frac)))
        call_llm_this_batch = batch_idx < taper_cutoff

    # ── Generate LLM candidates (skipped once past the taper cutoff) ──
    llm_fallback_used, llm_retry_count = False, 0
    if call_llm_this_batch:
        llm_forms, llm_meta = _llm_generate_candidates(
            X_obs, Y_obs, prior_text, n_llm_candidates,
            model, mock_llm, rng,
        )
        llm_fallback_used = llm_meta["llm_fallback_used"]
        llm_retry_count = llm_meta["llm_retry_count"]
    else:
        llm_forms = []

    # Convert to vectors
    llm_vectors = np.array([formulation_to_vector(f) for f in llm_forms]) \
                  if llm_forms else np.zeros((0, d))

    # Within-batch deduplication: diversity_select keeps only the most
    # mutually-diverse subset of the LLM's own proposals before they ever
    # reach the acquisition scorer, instead of only filtering against prior
    # observations. Near-duplicate LLM proposals within a single batch were
    # a measured problem elsewhere in this codebase (strategy_mo_llm in
    # excipient_campaign_mo.py) — this applies the same fix here.
    if len(llm_vectors) > 0:
        from llm_warmstart import diversity_select
        existing_n = (X_obs - lo) / (hi - lo + 1e-12) if len(X_obs) > 0 else None
        n_keep = max(1, len(llm_vectors) // 2)  # keep the most diverse half
        dedup_idx = diversity_select(
            (llm_vectors - lo) / (hi - lo + 1e-12), n_keep, existing_n=existing_n)
        llm_vectors = llm_vectors[dedup_idx]

    # Remove near-duplicates of existing observations
    if len(X_obs) > 0 and len(llm_vectors) > 0:
        X_obs_n = (X_obs - lo) / (hi - lo + 1e-12)
        llm_n = (llm_vectors - lo) / (hi - lo + 1e-12)
        keep = []
        for i in range(len(llm_n)):
            if np.min(np.linalg.norm(X_obs_n - llm_n[i], axis=1)) > 0.05:
                keep.append(i)
        llm_vectors = llm_vectors[keep] if keep else llm_vectors

    # ── Generate EGBO candidates (evolutionary) ──
    try:
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement)
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import SumMarginalLogLikelihood
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.transforms import normalize, unnormalize
        from botorch.optim.optimize import optimize_acqf
        from pymoo.algorithms.moo.unsga3 import UNSGA3
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.core.termination import NoTermination
        from pymoo.util.ref_dirs import get_reference_directions

        tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
        M = Y_obs.shape[1]
        evo_candidates = 20

        Y_int = Y_obs.copy()
        for j, direction in enumerate(OBJECTIVE_DIRECTIONS):
            if direction == "min":
                Y_int[:, j] = -Y_int[:, j]

        X_norm = (X_obs - lo) / (hi - lo + 1e-12)
        train_x = torch.tensor(X_norm, **tkwargs)
        train_y = torch.tensor(Y_int, **tkwargs)
        standard_bounds = torch.zeros(2, d, **tkwargs)
        standard_bounds[1] = 1.0

        Y_all_int = oracle._Y_raw.copy()
        for j, direction in enumerate(OBJECTIVE_DIRECTIONS):
            if direction == "min":
                Y_all_int[:, j] = -Y_all_int[:, j]
        ref_point = torch.tensor(
            Y_all_int.min(axis=0) -
            0.1 * (Y_all_int.max(axis=0) - Y_all_int.min(axis=0) + 1e-9),
            **tkwargs)

        models = [SingleTaskGP(train_x, train_y[:, j:j+1],
                               outcome_transform=Standardize(m=1))
                 for j in range(M)]
        model_gp = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model_gp.likelihood, model_gp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll, max_attempts=1)

        acq_fn = qLogNoisyExpectedHypervolumeImprovement(
            model=model_gp, ref_point=ref_point, X_baseline=train_x,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
            prune_baseline=True, cache_root=True,
        )

        # Acquisition-optimised candidates
        try:
            qbo_x, _ = optimize_acqf(
                acq_fn, bounds=standard_bounds, q=batch_size,
                num_restarts=2, raw_samples=16,
                options={"maxiter": 20},
            )
        except Exception:
            qbo_x = torch.rand(batch_size, d, **tkwargs)

        # Evolutionary candidates
        pf_idx_np = pareto_front_of(Y_obs)
        seed_pool_idx = pf_idx_np if len(pf_idx_np) > 0 else np.arange(len(Y_obs))
        seed_x = train_x[seed_pool_idx].cpu().numpy()
        if seed_x.shape[0] < evo_candidates:
            pad = rng.random((evo_candidates - seed_x.shape[0], d))
            seed_x = np.vstack([seed_x, pad])
        seed_x = seed_x[:max(evo_candidates, 2)]

        try:
            ref_dirs = get_reference_directions("energy", M, evo_candidates,
                                                seed=int(rng.integers(1e6)))
            pop_size = max(len(ref_dirs), evo_candidates, 2)
            algo = UNSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=seed_x)
            pm = PymooProblem(n_var=d, n_obj=M, n_constr=0,
                              xl=np.zeros(d), xu=np.ones(d))
            algo.setup(pm, termination=NoTermination())
            pop = algo.ask()
            pop_size_actual = len(pop)
            f_idx = seed_pool_idx[:pop_size_actual] if \
                len(seed_pool_idx) >= pop_size_actual else seed_pool_idx
            f_vals = -Y_int[f_idx]
            if len(f_vals) >= pop_size_actual:
                pop.set("F", f_vals[:pop_size_actual])
            else:
                pad_f = np.tile(f_vals[-1:], (pop_size_actual - len(f_vals), 1))
                pop.set("F", np.vstack([f_vals, pad_f]))
            algo.tell(infills=pop)
            ea_x = torch.tensor(algo.ask().get("X"), **tkwargs)
        except Exception:
            ea_x = torch.rand(evo_candidates, d, **tkwargs)

        # ── Merge LLM + EGBO candidates ──
        egbo_cands_n = torch.cat([qbo_x, ea_x], dim=0).cpu().numpy()

        if len(llm_vectors) > 0:
            llm_n = (llm_vectors - lo) / (hi - lo + 1e-12)
            all_cands_n = np.vstack([egbo_cands_n, llm_n])
        else:
            all_cands_n = egbo_cands_n

        all_cands_t = torch.tensor(all_cands_n, **tkwargs)

        # Score all candidates with qLogNEHVI in a single batched call
        # (all_cands_t.unsqueeze(1) -> shape (N, 1, d), N independent q=1
        # evaluations) instead of looping one candidate at a time — same
        # vectorisation fix already applied in strategy_ls_na_egbo.py.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                acq_vals = acq_fn(all_cands_t.unsqueeze(1)).detach().cpu().numpy()
            except Exception:
                acq_vals = []
                for i in range(all_cands_t.shape[0]):
                    try:
                        v = float(acq_fn(all_cands_t[i].unsqueeze(0)).item())
                    except Exception:
                        v = float("-inf")
                    acq_vals.append(v)
                acq_vals = np.array(acq_vals)

        # Novelty-aware selection
        X_obs_n = X_norm
        selected_idx = novelty_aware_select_vectorised(
            all_cands_n, acq_vals, batch_size,
            w_acq=w_acq, w_nov=w_nov,
            X_obs_n=X_obs_n,
        )

        new_x_norm = all_cands_n[selected_idx]
        new_x_raw = (unnormalize(torch.tensor(new_x_norm, **tkwargs),
                     torch.tensor(np.column_stack([lo, hi]).T, **tkwargs))
                     .cpu().numpy())

        n_llm_in_pool = len(llm_vectors)
        n_egbo_in_pool = len(egbo_cands_n)

        return new_x_raw, {
            "novelty_select": True,
            "n_llm_candidates": n_llm_in_pool,
            "n_egbo_candidates": n_egbo_in_pool,
            "n_total_candidates": len(all_cands_n),
            "w_acq": w_acq,
            "w_nov": w_nov,
            "llm_called_this_batch": call_llm_this_batch,
            "llm_fallback_used": llm_fallback_used,
            "llm_retry_count": llm_retry_count,
        }

    except Exception as e:
        warnings.warn(f"strategy_mo_llm_candidate_gen failed ({e}), "
                      f"falling back to lightweight strategy_mo_egbo.")
        cands, extra = strategy_mo_egbo(oracle, X_obs, Y_obs, bounds,
                                        batch_size, rng, **kw)
        extra["llm_called_this_batch"] = call_llm_this_batch
        extra["llm_fallback_used"] = llm_fallback_used
        extra["llm_retry_count"] = llm_retry_count
        extra["fallback_reason"] = str(e)
        return cands, extra
