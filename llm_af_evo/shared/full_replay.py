"""
full_replay.py — 2b (full counterfactual replay) infrastructure: run a real
sequential BO loop where the EVOLVED AF (not qLogNEHVI/novelty-aware
selection) picks each batch, against the real oracle.

strategy_evolved_af is adapted from strategy_ls_na_egbo.py's
strategy_mo_egbo_novelty: candidate GENERATION (GP fit, qLogNEHVI-optimised
candidates via optimize_acqf, U-NSGA-III evolutionary candidates) is kept
IDENTICAL — that's what produces the same kind of candidate pool the
evolved AFs were trained on, and it's also the expensive part (GP fit +
optimize_acqf's internal repeated acquisition evaluation), so it cannot be
skipped without evaluating the AF on a differently-distributed pool than
what it saw during training. What's REPLACED is only the final scoring
step: instead of qLogNEHVI's own batched acq_fn(candidates) call plus
novelty_aware_select_vectorised, the evolved AF is called via the sandbox
(same af_interface.py contract as training) and its top-batch_size scores
are used directly (af_interface.select_batch).

This file exists specifically to MEASURE the real cost of 2b before
committing to an evolution run under it — the naive "swap qLogNEHVI's
scoring for numpy, should be ~free" argument turned out to ignore that
optimize_acqf's internal cost (not the final scoring pass) is what
generate_training_set.py's 6.5s/batch timing pilot actually measured, and
that part is NOT removable. See run_2b_diagnostic.py for the actual
timing + ranking-comparison measurements this file supports.

FIXED (previously a known simplification): stagnant_batches is now tracked
correctly across a full campaign via strategy_evolved_af's optional
hv_history parameter — see that function's docstring. run_mo_campaign
itself still doesn't expose its running hv_trajectory to strategy_fn (that
would require changing excipient_campaign_mo.py, used far beyond this
file), so the fix instead threads a mutable list through strategy_kwargs,
which run_mo_campaign's per-batch loop reuses (not recopies) across calls —
run_2b_campaign now passes a fresh hv_history=[] per campaign so evolved
AFs conditioning on stagnant_batches (e.g. phase_decaying_ucb) are finally
scored under the real per-batch HV trajectory during 2b training/
validation, not a constant 0. Every OTHER caller of strategy_evolved_af
across the repo (run_batch_size_ablation.py, run_coatings_generalization.py,
run_mc_hvi_pilot.py, run_unsga3_pool_pilot.py, composition_pilot_common.py,
etc.) does not pass hv_history and is unaffected — the parameter defaults
to None, which reproduces the exact old always-0 behaviour, so none of
those call sites needed to change.
"""

import pathlib
import sys
import warnings

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# torch.manual_seed(seed) alone does NOT make CPU runs reproducible: GP
# fitting's Cholesky decomposition, qLogNEHVI's MC sampling, and scipy's
# L-BFGS-B inside optimize_acqf all touch multi-threaded BLAS operations
# whose floating-point reduction order depends on thread scheduling, not
# the RNG seed. Verified directly — two nominally-identical 20-campaign
# coatings runs (same --seed, same code) produced p=0.033 (ehvi_approx
# "significantly" beating baseline) and p=0.82 (null) respectively, a swing
# far too large to be genuine sampling variance from re-running the exact
# same experiment. Forcing single-threaded execution is the standard fix
# for CPU BLAS non-determinism. This is set once, at import time, so every
# script that imports full_replay (run_coatings_generalization.py,
# evolve_af_2b.py, validate_2b_seed.py, validate_population_2b.py,
# run_2b_diagnostic.py) gets it automatically — it's a process-wide
# setting, not something that can be scoped per-call.
torch.set_num_threads(1)

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

from excipient_oracle_mo import DiscreteMOExcipientOracle
from excipient_campaign_mo import (
    run_mo_campaign, pareto_front_of, OBJECTIVE_DIRECTIONS,
)
from strategy_ls_na_egbo import strategy_mo_egbo, strategy_mo_egbo_novelty

from sandbox import run_af_in_sandbox, SandboxError
from af_interface import select_batch, SEED_PROGRAMS
from gen_sandbox import run_generator_in_sandbox, GenSandboxError
from fitness_common import to_allmax, hv_of
from ada_coatings_oracle import (
    DiscreteADACoatingsOracle, FEATURE_DIM as _COATINGS_FEATURE_DIM,
    OBJECTIVE_NAMES as _COATINGS_OBJECTIVE_NAMES,
)
from tunable_synthetic_oracle import TunableSyntheticMOOracle

# Opt-in diagnostic log for the growth-aware modification (1)'s
# boundary_std / front_range / growth_weight per batch — off (None) by
# default so ordinary callers pay zero cost. Enable via
# enable_boundary_debug_log(), tag each campaign via
# set_boundary_debug_tag(...) so rows can be traced back to
# (domain_seed, campaign) by a caller that only sees campaign-level
# granularity, then read back via get_boundary_debug_log().
_BOUNDARY_DEBUG_LOG = None
_BOUNDARY_DEBUG_TAG = None


def enable_boundary_debug_log():
    global _BOUNDARY_DEBUG_LOG
    _BOUNDARY_DEBUG_LOG = []


def set_boundary_debug_tag(tag):
    global _BOUNDARY_DEBUG_TAG
    _BOUNDARY_DEBUG_TAG = tag


def get_boundary_debug_log():
    return _BOUNDARY_DEBUG_LOG


# Same pattern, for modification (1b)/v3's pool-based growth signal: the
# std of whichever pool candidate currently has the highest predicted
# mean for each objective (an unobserved point, unlike v1's boundary
# point) — see run_growth_aware_v3.py's module docstring.
_POOL_TOP_DEBUG_LOG = None


def enable_pool_top_debug_log():
    global _POOL_TOP_DEBUG_LOG
    _POOL_TOP_DEBUG_LOG = []


def get_pool_top_debug_log():
    return _POOL_TOP_DEBUG_LOG


# Expected oracle_X_raw/oracle_Y_raw column counts per oracle_family — used
# only to fail loudly (ValueError) instead of silently misbehaving when a
# training log from the wrong oracle family gets reconstructed as if it
# were the other one (e.g. --train_dir pointing at excipient logs while
# --oracle coatings was passed to evolve_af_v2.py) — this exact mismatch
# is easy to make since neither evolve_af_v2.py's --train_dir default nor
# a training log's own contents advertise which oracle family produced
# them.
_ORACLE_SHAPE_EXPECTATIONS = {
    "excipient": {"feature_dim": 16, "n_objectives": 3},  # Tm, kD, viscosity
    "coatings": {"feature_dim": _COATINGS_FEATURE_DIM,
                 "n_objectives": len(_COATINGS_OBJECTIVE_NAMES)},
    "tunable": {"feature_dim": 6, "n_objectives": 2},  # f1, f2 — see
                 # tunable_synthetic_oracle.py's TunableSyntheticMOOracle.build
                 # default d=6; override _ORACLE_SHAPE_EXPECTATIONS here (or
                 # pass a matching d) if a v3 training-set generator uses a
                 # different dimensionality.
}

# TunableSyntheticMOOracle's noise/plateau/scale config (see its own
# docstring) is NOT recoverable from a training log's plain X_raw/Y_raw
# arrays the way excipient/coatings' fixed real data is — it must travel
# WITH the log. v3's training-set generator (not yet written) needs to dump
# these fields into every log JSON it produces (alongside the usual
# X_init/Y_init/oracle_X_raw/oracle_Y_raw/budget/n_init/batch_size keys):
# tunable_scale1, tunable_scale2, tunable_noise_level, tunable_noise_mode,
# tunable_boundary_gain, tunable_seed. reconstruct_oracle below reads them
# with the TunableSyntheticMOOracle.build() defaults as a fallback ONLY so
# a hand-built log without these keys doesn't crash — for a real v3 run
# generate them explicitly and dump them, don't rely on this fallback
# silently picking generic defaults that may not match what was intended.
_TUNABLE_DEFAULTS = dict(scale1=1.0, scale2=3.0, noise_level=0.08,
                          noise_mode="proportional", boundary_gain=0.5, seed=0)


def reconstruct_oracle(log: dict, oracle_family: str = "excipient"):
    """
    Rebuild a queryable oracle from a held-out log's dumped
    oracle_X_raw/oracle_Y_raw. Raises ValueError if the log's column
    counts don't match oracle_family's expected shape (see
    _ORACLE_SHAPE_EXPECTATIONS) — this catches a wrong --train_dir/
    --oracle pairing immediately, rather than reconstructing a
    dimensionally-nonsensical oracle and failing confusingly (or not at
    all) much later inside GP fitting.

    oracle_family: "excipient" (default, backward-compatible with every
    existing caller — evolve_af_2b.py, validate_2b_seed.py,
    validate_population_2b.py, run_2b_diagnostic.py all rely on this
    default and don't pass this parameter) rebuilds a
    DiscreteMOExcipientOracle, refitting its StandardScaler from
    oracle_X_raw alone (a pure function of X_raw, exactly how
    DiscreteMOExcipientOracle.build originally constructed it) — forms
    isn't needed by anything this replay uses (bounds, objective_
    directions/names, query_mo), so it's passed as None.

    "coatings" rebuilds a DiscreteADACoatingsOracle instead — its
    constructor takes just (X_raw, Y_raw) and builds its own scaler
    internally, so no forms/scaler plumbing is needed for that branch.
    Both classes expose the identical interface (bounds(),
    objective_directions(), objective_names(), query_mo(), and the
    _X_raw/_Y_raw/_scaler/_queried attributes) this replay's shared
    campaign-running code (run_mo_campaign, strategy_evolved_af) expects —
    see DiscreteADACoatingsOracle's own docstring, which states this
    compatibility explicitly.
    """
    X_raw = np.array(log["oracle_X_raw"])
    Y_raw = np.array(log["oracle_Y_raw"])

    expected = _ORACLE_SHAPE_EXPECTATIONS.get(oracle_family)
    if expected is not None and (X_raw.shape[1] != expected["feature_dim"]
                                  or Y_raw.shape[1] != expected["n_objectives"]):
        raise ValueError(
            f"reconstruct_oracle(oracle_family={oracle_family!r}) expected "
            f"oracle_X_raw with {expected['feature_dim']} columns and "
            f"oracle_Y_raw with {expected['n_objectives']} columns, but this "
            f"training log has {X_raw.shape[1]} and {Y_raw.shape[1]} — this "
            f"log looks like it came from a different oracle family. Check "
            f"that --train_dir/--heldout_dir actually points at logs "
            f"generated for --oracle {oracle_family!r} (generate_training_set.py "
            f"for 'excipient', generate_coatings_training_set.py for 'coatings').")

    if oracle_family == "coatings":
        return DiscreteADACoatingsOracle(X_raw, Y_raw)
    if oracle_family == "tunable":
        # CAVEAT (see _TUNABLE_DEFAULTS's comment above): TunableSyntheticMOOracle
        # normally DRAWS its own noisy _Y_raw from _Y_true + a fresh RNG draw
        # inside __init__ (tunable_synthetic_oracle.py:157-158) — it does not
        # accept a pre-realized noisy Y_raw the way DiscreteADACoatingsOracle/
        # DiscreteMOExcipientOracle do for real, already-fixed experimental
        # data. Passing this log's dumped oracle_Y_raw as Y_true here means
        # the reconstructed oracle will draw a DIFFERENT noise realization
        # than whatever the training-set generator originally saw for this
        # campaign, even with the same seed/config — replay determinism for
        # this oracle family is NOT yet guaranteed the way it is for
        # excipient/coatings. Treat this branch as provisional until a v3
        # training-set generator + a matching reconstruct fix (e.g. an
        # alternate constructor that accepts a fixed Y_raw directly) exists;
        # don't trust cross-run comparisons on "tunable" until then.
        cfg = {**_TUNABLE_DEFAULTS, **{k[len("tunable_"):]: v for k, v in log.items()
                                        if k.startswith("tunable_")}}
        return TunableSyntheticMOOracle(
            X_raw, Y_raw, objective_names=["f1", "f2"],
            scale1=cfg["scale1"], scale2=cfg["scale2"],
            noise_level=cfg["noise_level"], noise_mode=cfg["noise_mode"],
            boundary_gain=cfg["boundary_gain"], seed=cfg["seed"])
    scaler = StandardScaler().fit(X_raw)
    return DiscreteMOExcipientOracle(X_raw, Y_raw, forms=None, scaler=scaler)


def strategy_evolved_af(oracle, X_obs, Y_obs, bounds, batch_size, rng,
                         af_code: str, budget: int, sandbox_log_dir=None,
                         evo_candidates: int = 20, hv_history: list = None,
                         n_init: int = None, **kw):
    """
    Candidate generation identical to strategy_mo_egbo_novelty (see module
    docstring). Final selection: the evolved AF via the sandbox, not
    qLogNEHVI + novelty-aware selection.

    hv_history: optional MUTABLE list, shared by the caller across every
    batch of one campaign (via the same strategy_kwargs dict run_mo_campaign
    reuses unchanged across its per-batch loop), used to compute
    stagnant_batches correctly instead of hardcoding 0 — see module
    docstring's "FIXED" note. Defaults to None, in which case a fresh local
    list is used (no cross-call persistence), reproducing the exact old
    always-0 behaviour for every caller that doesn't pass this explicitly.

    At call b (0-indexed) within one campaign, Y_obs reflects the seed
    front plus every batch's outcome strictly BEFORE b — run_mo_campaign
    calls strategy_fn with X_obs/Y_obs as they stood before appending the
    batch this call itself is about to choose. So this call's own front HV
    (current_hv, from front_allmax below) equals batch (b-1)'s outcome for
    b>=1, or just the seed front's HV for b=0 (not any batch's outcome at
    all). Matching evolve_af.py's _stagnant_prefix semantics (which counts
    non-improving transitions over REAL batch outcomes only, never the
    seed front, and never this decision's own not-yet-known result) means
    the seed front's HV must never be treated as if it were a batch
    outcome — a leading None sentinel in hv_history marks "the very first
    call for this campaign already happened" without counting its value.
    """
    try:
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement)
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import SumMarginalLogLikelihood
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.transforms import unnormalize
        from botorch.optim.optimize import optimize_acqf
        from pymoo.algorithms.moo.unsga3 import UNSGA3
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.core.termination import NoTermination
        from pymoo.util.ref_dirs import get_reference_directions

        tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
        lo, hi = bounds[:, 0], bounds[:, 1]
        d = bounds.shape[0]
        M = Y_obs.shape[1]
        # Oracle-provided directions, not the module-level OBJECTIVE_DIRECTIONS
        # constant — see excipient_campaign_mo.run_mo_campaign's identical fix.
        directions = oracle.objective_directions()

        Y_int = Y_obs.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_int[:, j] = -Y_int[:, j]

        X_norm = (X_obs - lo) / (hi - lo + 1e-12)
        train_x = torch.tensor(X_norm, **tkwargs)
        train_y = torch.tensor(Y_int, **tkwargs)
        standard_bounds = torch.zeros(2, d, **tkwargs)
        standard_bounds[1] = 1.0

        Y_all_int = oracle._Y_raw.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_all_int[:, j] = -Y_all_int[:, j]
        ref_point = torch.tensor(
            Y_all_int.min(axis=0) -
            0.1 * (Y_all_int.max(axis=0) - Y_all_int.min(axis=0) + 1e-9),
            **tkwargs)

        models = [SingleTaskGP(train_x, train_y[:, j:j + 1],
                                outcome_transform=Standardize(m=1))
                  for j in range(M)]
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll, max_attempts=1)

        acq_fn = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=ref_point, X_baseline=train_x,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
            prune_baseline=True, cache_root=True,
        )

        try:
            qbo_x, _ = optimize_acqf(
                acq_fn, bounds=standard_bounds, q=batch_size,
                num_restarts=2, raw_samples=16,
                options={"maxiter": 20},
            )
        except Exception:
            qbo_x = torch.rand(batch_size, d, **tkwargs)

        pf_idx_np = pareto_front_of(Y_obs, directions=directions)
        seed_pool_idx = pf_idx_np if len(pf_idx_np) > 0 else np.arange(len(Y_obs))
        seed_x = train_x[seed_pool_idx].cpu().numpy()
        if seed_x.shape[0] < evo_candidates:
            pad = rng.random((evo_candidates - seed_x.shape[0], d))
            seed_x = np.vstack([seed_x, pad])
        seed_x = seed_x[:max(evo_candidates, 2)]

        # Growth-aware modification (1): GP posterior std at each
        # objective's own front boundary point (the observed pareto-front
        # point that currently maximises that objective, in all-maximise
        # convention). Unlike "does the candidate pool contain a point
        # exceeding front_max" (which qLogNEHVI's own candidate pipeline
        # essentially never produces, since it fills hypervolume gaps
        # within the observed front rather than proposing single-
        # objective extrapolations — see run_growth_aware.py's module
        # docstring), this signal is a continuous quantity available
        # every batch: how uncertain the GP still is right at the edge of
        # what's been explored for that objective. High uncertainty there
        # means the GP hasn't ruled out further extension, so it fires
        # during real exploration instead of being gated by a candidate-
        # pipeline behaviour.
        front_boundary_std = {}
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            pf_x_np = train_x[seed_pool_idx].cpu().numpy()
            pf_y_int = Y_int[seed_pool_idx]
            for j, name in enumerate(oracle.objective_names()):
                j_boundary_local = int(np.argmax(pf_y_int[:, j]))
                x_boundary = torch.tensor(
                    pf_x_np[j_boundary_local:j_boundary_local + 1], **tkwargs)
                post_b = model.posterior(x_boundary)
                std_b = post_b.variance.clamp_min(1e-12).sqrt().detach().cpu().numpy()
                front_boundary_std[name] = float(std_b[0, j])

        if _BOUNDARY_DEBUG_LOG is not None:
            # front_range as the sandbox itself computes it (range over ALL
            # observed points in all-maximise convention, not just the
            # non-dominated subset — matches sandbox.py's pareto_front_range,
            # which is keyed off front_allmax = to_allmax(Y_obs)).
            for j, name in enumerate(oracle.objective_names()):
                front_range_j = max(
                    float(Y_int[:, j].max() - Y_int[:, j].min()), 1e-6)
                b_std = front_boundary_std[name]
                _BOUNDARY_DEBUG_LOG.append({
                    "tag": _BOUNDARY_DEBUG_TAG,
                    "n_obs": len(X_obs),
                    "objective": name,
                    "boundary_std": b_std,
                    "front_range": front_range_j,
                    "growth_weight": 1.0 + b_std / front_range_j,
                })

        try:
            ref_dirs = get_reference_directions("energy", M, evo_candidates,
                                                 seed=int(rng.integers(1e6)))
            pop_size = max(len(ref_dirs), evo_candidates, 2)
            algo = UNSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=seed_x,
                          seed=int(rng.integers(1e6)))
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
                pad = np.tile(f_vals[-1:], (pop_size_actual - len(f_vals), 1))
                pop.set("F", np.vstack([f_vals, pad]))
            algo.tell(infills=pop)
            ea_x = torch.tensor(algo.ask().get("X"), **tkwargs)
        except Exception:
            ea_x = torch.rand(evo_candidates, d, **tkwargs)

        candidates = torch.cat([qbo_x, ea_x], dim=0)
        candidates_n = candidates.cpu().numpy()

        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            post = model.posterior(candidates)
            pool_mu = post.mean.detach().cpu().numpy()
            pool_sigma = post.variance.clamp_min(1e-12).sqrt().detach().cpu().numpy()

        if _POOL_TOP_DEBUG_LOG is not None:
            for j, name in enumerate(oracle.objective_names()):
                front_range_j = max(
                    float(Y_int[:, j].max() - Y_int[:, j].min()), 1e-6)
                top_idx = int(np.argmax(pool_mu[:, j]))
                top_std = float(pool_sigma[top_idx, j])
                _POOL_TOP_DEBUG_LOG.append({
                    "tag": _BOUNDARY_DEBUG_TAG,
                    "n_obs": len(X_obs),
                    "objective": name,
                    "top_candidate_std": top_std,
                    "front_range": front_range_j,
                    "growth_weight": 1.0 + top_std / front_range_j,
                })

        front_allmax = to_allmax(Y_obs.copy(), directions=directions)
        # n_init opt-in: the front AT INIT ONLY, for AFs wanting a frozen
        # (non-growing) normalisation denominator — see sandbox.py's
        # front_allmax_init docstring. Y_obs's first n_init rows are always
        # the campaign's init points regardless of which batch this call
        # is for (run_mo_campaign only ever appends to X_obs/Y_obs).
        front_allmax_init = (to_allmax(Y_obs[:n_init].copy(), directions=directions)
                              if n_init else None)
        ref_point_allmax = ref_point.cpu().numpy()
        step = len(X_obs)
        # See this function's docstring / module docstring's "FIXED" note.
        # Y_obs at THIS call reflects every batch's outcome strictly BEFORE
        # this one — so current_hv (this call's own front HV) equals the
        # PREVIOUS batch's completed outcome, except on the very first call
        # of a campaign, where it's just the seed front's HV (not any
        # batch's outcome at all — evolve_af.py's hv_trajectory/
        # _stagnant_prefix never counts the seed front, only real batch
        # outcomes, so counting it here would shift stagnant_batches one
        # step earlier than what training ever saw). A leading None
        # sentinel marks "this campaign's very first call already
        # happened" without polluting the real HV-outcome history with the
        # seed front's value.
        if hv_history is None:
            hv_history = []
        current_hv = hv_of(front_allmax, ref_point_allmax)
        is_first_call = len(hv_history) == 0
        past_batch_hvs = [v for v in hv_history if v is not None]
        history_for_decision = [] if is_first_call else past_batch_hvs + [current_hv]
        stagnant_batches = (
            sum(1 for i in range(1, len(history_for_decision))
                if history_for_decision[i] <= history_for_decision[i - 1] + 1e-6)
            if len(history_for_decision) >= 2 else 0)
        hv_history.append(None if is_first_call else current_hv)

        scores = run_af_in_sandbox(
            af_code, candidates_n, pool_mu, pool_sigma, X_norm,
            front_allmax, ref_point_allmax, step, budget, step, stagnant_batches,
            front_allmax,
            objective_names=oracle.objective_names(),
            log_dir=sandbox_log_dir,
            front_boundary_std=front_boundary_std,
            front_allmax_init=front_allmax_init,
        )
        selected_idx = select_batch(scores, batch_size)

        new_x_norm = candidates[selected_idx]
        new_x_raw = (unnormalize(new_x_norm, torch.tensor(
            np.column_stack([lo, hi]).T, **tkwargs)).cpu().numpy())

        return new_x_raw, {"evolved_af": True, "n_candidates": len(candidates)}

    except SandboxError as e:
        # A sandbox failure here means the AF itself is broken for this
        # input, not that candidate generation failed — surface it rather
        # than silently falling back, since that would corrupt the timing/
        # ranking measurement this file exists to produce.
        raise
    except Exception as e:
        warnings.warn(f"strategy_evolved_af candidate generation failed ({e}), "
                       f"falling back to lightweight strategy_mo_egbo.")
        cands, extra = strategy_mo_egbo(oracle, X_obs, Y_obs, bounds, batch_size, rng)
        extra["evolved_af"] = False
        extra["fallback_reason"] = str(e)
        return cands, extra


def pool_obj_correlation(model, candidates, names, use_da_coreg: bool) -> dict:
    """
    Per-candidate cross-objective posterior correlation, keyed by
    "name_a,name_b" (JSON has no tuple keys) -> list[float], one entry per
    candidate. Only ever populated for the DA-COREG surrogate — a
    ModelListGP's per-objective models are fit independently, so their
    cross-objective covariance is trivially zero and not worth threading
    through the AF sandbox for the independent-GP path (returns {} there).

    Computed via one posterior() call per candidate rather than slicing the
    single batched pool posterior (post = model.posterior(candidates))
    strategy_unsga3_pool_af already computes for pool_mu/pool_sigma:
    MultiTaskGP's batched-q covariance_matrix is the JOINT covariance across
    BOTH candidates and tasks, not a block-diagonal stack of per-candidate
    MxM blocks — indexing into it for "candidate i's own MxM block" is
    layout-dependent and easy to get silently wrong. Re-querying the
    posterior one candidate at a time sidesteps that ambiguity entirely, at
    the cost of len(candidates) extra (cheap relative to the GP fit itself)
    posterior evaluations, and is only done when use_da_coreg=True.
    """
    M = len(names)
    if not use_da_coreg or M < 2:
        return {}
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    out = {f"{a},{b}": [] for a, b in pairs}
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        for i in range(candidates.shape[0]):
            try:
                cov = (model.posterior(candidates[i:i + 1]).mvn.covariance_matrix
                       .detach().cpu().numpy().reshape(M, M))
                for a, b in pairs:
                    ia, ib = names.index(a), names.index(b)
                    denom = np.sqrt(max(cov[ia, ia], 1e-12) * max(cov[ib, ib], 1e-12))
                    out[f"{a},{b}"].append(float(cov[ia, ib] / denom) if denom > 0 else 0.0)
            except Exception:
                for a, b in pairs:
                    out[f"{a},{b}"].append(0.0)
    return out


def strategy_unsga3_pool_af(oracle, X_obs, Y_obs, bounds, batch_size, rng,
                             af_code: str, budget: int, sandbox_log_dir=None,
                             evo_candidates: int = 25, use_da_coreg: bool = False,
                             use_front_range_norm: bool = False, **kw):
    """
    The missing cell in the baseline-decomposition table: scores/selects
    over a candidate pool generated PURELY by UNSGA3 — no qLogNEHVI-
    gradient-optimized qbo_x mixed in at all — using simple per-candidate
    scoring (af_code, e.g. trust_only/ehvi_approx) and top-k selection.

    use_da_coreg swaps the independent per-objective ModelListGP for
    compose_strategies._fit_model's coregionalized MultiTaskGP surrogate,
    with everything else (UNSGA3-only candidates, af_code scoring,
    select_batch) held fixed. This is the one qLogNEHVI-free cell DA-COREG
    was never tested in: run_da_coreg_pilot.py's own ablation kept
    qLogNEHVI-optimized qbo_x in the candidate pool for every cell
    (deliberately, to hold candidate distribution fixed across THAT
    ablation) and always scored via qLogNEHVI itself, so its DTLZ2 loss
    (-3.8%, p=1.9e-6) can't distinguish "DA-COREG is bad" from "DA-COREG's
    posterior is bad specifically when consumed by qLogNEHVI's correlated
    joint MC batch scoring." Comparing unsga3_pool_af_indep vs
    unsga3_pool_af_da_coreg (both this function, only use_da_coreg
    differs) isolates that.

    use_front_range_norm (only meaningful when use_da_coreg=True; a no-op
    otherwise) rescales each objective's train_y by its current Pareto
    front's range BEFORE fitting DA-COREG's jointly-shared task-covariance
    kernel, then rescales model.posterior()'s mean/variance back to raw
    units immediately after, so nothing downstream of the fit (af_code
    scoring, front_allmax, etc.) sees anything but original-scale numbers.
    This isolates §22's [33]-motivated pitfall: da_coreg.py's
    fit_da_coreg_model stacks all M objectives into one flat column and
    applies a single global Standardize(m=1)) over that concatenation, so
    if e.g. viscosity's raw scale dwarfs Tm's, the learned cross-task
    correlation can be dominated by whichever objective has the largest raw
    range before any acquisition ever sees it. Independent GPs already fit
    one Standardize per objective and are unaffected either way — this flag
    changes DA-COREG's fit inputs only, nothing else, so a difference is
    attributable to this one variable (see docs/llm_evolved_afs_
    comprehensive_log.md §22 "Execution plan", step 1).

    Not yet exercised end-to-end against a real botorch install in this
    environment — the same caveat da_coreg.py's own docstring carries.

    Isolates something the existing ladder structurally can't: whether the
    gradient-optimized candidates in the usual qbo_x+ea_x pool are
    contributing anything beyond what UNSGA3 diversity alone provides.
    qnehvi_plain drops UNSGA3 but keeps gradient candidates AND qLogNEHVI's
    own scoring; unsga3_plain drops gradient candidates AND GP-based
    scoring simultaneously — neither isolates "UNSGA3-only pool, GP-based
    scoring." If this condition lands in the same +0.9%..+2.4% band the
    evolved-AF/mo_egbo_real cluster occupies, that's evidence UNSGA3
    diversity alone is doing all the work and gradient-optimized candidates
    are dispensable too, sharpening "joint gradient optimization AND
    evolutionary diversity, both needed" down to "evolutionary diversity,
    full stop." If it drops toward qnehvi_plain/unsga3_plain territory
    instead, the gradient-optimized candidates ARE contributing something
    real, and the two-ingredient story holds.

    evo_candidates=25 (not the usual 20) so the UNSGA3-only pool matches
    the usual qbo_x+ea_x pool's TOTAL size (5+20=25) — pool composition is
    the thing being isolated here, not pool size.

    Still requires a GP fit: scoring (trust_only/ehvi_approx) reads
    cand["gp_posterior"], which needs a fitted model queried at wherever
    the candidates ended up, regardless of how those candidates were
    generated — the GP is part of the SCORING step, not "generation," in
    this decomposition.
    """
    try:
        from botorch.utils.transforms import unnormalize
        from pymoo.algorithms.moo.unsga3 import UNSGA3
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.core.termination import NoTermination
        from pymoo.util.ref_dirs import get_reference_directions

        # Local import: full_replay.py mustn't require compose_strategies'
        # (and thus da_coreg's) dependencies to be importable when
        # use_da_coreg=False, matching this function's existing pattern of
        # only importing what a given call path actually needs.
        from compose_strategies import _fit_model

        tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
        lo, hi = bounds[:, 0], bounds[:, 1]
        d = bounds.shape[0]
        M = Y_obs.shape[1]
        directions = oracle.objective_directions()

        Y_int = Y_obs.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_int[:, j] = -Y_int[:, j]

        X_norm = (X_obs - lo) / (hi - lo + 1e-12)
        train_x = torch.tensor(X_norm, **tkwargs)
        train_y = torch.tensor(Y_int, **tkwargs)

        # §22 step 1: rescale train_y by each objective's current Pareto
        # front range BEFORE fitting DA-COREG's jointly-shared task kernel,
        # so a raw-scale mismatch across objectives (e.g. viscosity >> Tm)
        # can't distort the learned cross-task correlation the way [33]
        # flags. Independent GPs (use_da_coreg=False) already fit one
        # Standardize() per objective and are unaffected by this flag
        # either way, so it's scoped to the DA-COREG branch only.
        frn_scale = None
        if use_da_coreg and use_front_range_norm:
            _pf_idx = pareto_front_of(Y_obs, directions=directions)
            pf_y = Y_int[_pf_idx] if len(_pf_idx) > 0 else Y_int
            frn_scale = np.clip(pf_y.max(axis=0) - pf_y.min(axis=0), 1e-9, None)
            train_y_fit = train_y / torch.tensor(frn_scale, **tkwargs)
        else:
            train_y_fit = train_y

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _fit_model(train_x, train_y_fit, use_da_coreg, tkwargs)

        pf_idx_np = pareto_front_of(Y_obs, directions=directions)
        seed_pool_idx = pf_idx_np if len(pf_idx_np) > 0 else np.arange(len(Y_obs))
        seed_x = X_norm[seed_pool_idx]
        if seed_x.shape[0] < evo_candidates:
            pad = rng.random((evo_candidates - seed_x.shape[0], d))
            seed_x = np.vstack([seed_x, pad])
        seed_x = seed_x[:max(evo_candidates, 2)]

        ref_dirs = get_reference_directions("energy", M, evo_candidates,
                                             seed=int(rng.integers(1e6)))
        pop_size = max(len(ref_dirs), evo_candidates, 2)
        algo = UNSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=seed_x,
                       seed=int(rng.integers(1e6)))
        pm = PymooProblem(n_var=d, n_obj=M, n_constr=0, xl=np.zeros(d), xu=np.ones(d))
        algo.setup(pm, termination=NoTermination())
        pop = algo.ask()
        pop_size_actual = len(pop)
        f_idx = seed_pool_idx[:pop_size_actual] if \
            len(seed_pool_idx) >= pop_size_actual else seed_pool_idx
        f_vals = -Y_int[f_idx]  # pymoo minimises
        if len(f_vals) >= pop_size_actual:
            pop.set("F", f_vals[:pop_size_actual])
        else:
            pad = np.tile(f_vals[-1:], (pop_size_actual - len(f_vals), 1))
            pop.set("F", np.vstack([f_vals, pad]))
        algo.tell(infills=pop)
        candidates = torch.tensor(algo.ask().get("X"), **tkwargs)
        candidates_n = candidates.cpu().numpy()

        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            post = model.posterior(candidates)
            pool_mu = post.mean.detach().cpu().numpy()
            pool_sigma = post.variance.clamp_min(1e-12).sqrt().detach().cpu().numpy()
        if frn_scale is not None:
            # Undo the fit-time rescale immediately — everything downstream
            # (af_code scoring, front_allmax, ref_point_allmax) must see
            # original-scale numbers, exactly as the use_front_range_norm=
            # False path always has. mean and std (already sqrt'd) both
            # scale linearly with frn_scale.
            pool_mu = pool_mu * frn_scale
            pool_sigma = pool_sigma * frn_scale

        # {} on the independent-GP path (use_da_coreg=False) — see
        # pool_obj_correlation's docstring for why.
        obj_correlation = pool_obj_correlation(model, candidates, oracle.objective_names(),
                                                use_da_coreg)

        front_allmax = to_allmax(Y_obs.copy(), directions=directions)
        Y_all_int = oracle._Y_raw.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_all_int[:, j] = -Y_all_int[:, j]
        ref_point_allmax = (Y_all_int.min(axis=0) -
                             0.1 * (Y_all_int.max(axis=0) - Y_all_int.min(axis=0) + 1e-9))

        step = len(X_obs)
        stagnant_batches = 0

        scores = run_af_in_sandbox(
            af_code, candidates_n, pool_mu, pool_sigma, X_norm,
            front_allmax, ref_point_allmax, step, budget, step, stagnant_batches,
            front_allmax,
            objective_names=oracle.objective_names(),
            log_dir=sandbox_log_dir,
            obj_correlation=obj_correlation,
        )
        selected_idx = select_batch(scores, batch_size)

        new_x_norm = candidates[selected_idx]
        new_x_raw = (unnormalize(new_x_norm, torch.tensor(
            np.column_stack([lo, hi]).T, **tkwargs)).cpu().numpy())

        return new_x_raw, {"unsga3_pool_af": True, "use_da_coreg": use_da_coreg,
                            "use_front_range_norm": use_front_range_norm,
                            "n_candidates": len(candidates),
                            "pool_x_norm": candidates_n.tolist(),
                            "pool_pred_mu": pool_mu.tolist(),
                            "pool_pred_sigma": pool_sigma.tolist(),
                            "pool_obj_correlation": obj_correlation}

    except SandboxError as e:
        raise
    except Exception as e:
        # Covers a DA-COREG MultiTaskGP fit/posterior failure too (it's
        # inside the same try block) — surfacing fallback_reason here is
        # what run_da_coreg_pilot.py's own fallback tracking exists for:
        # a silent fallback to strategy_mo_egbo must not look identical to
        # a genuine DA-COREG loss in a final_hv-only summary.
        warnings.warn(f"strategy_unsga3_pool_af failed ({e}), falling back to "
                       f"lightweight strategy_mo_egbo.")
        cands, extra = strategy_mo_egbo(oracle, X_obs, Y_obs, bounds, batch_size, rng)
        extra["unsga3_pool_af"] = False
        extra["use_da_coreg"] = use_da_coreg
        extra["fallback_reason"] = str(e)
        return cands, extra


def strategy_evolved_af_novelty(oracle, X_obs, Y_obs, bounds, batch_size, rng,
                                 af_code: str, budget: int, sandbox_log_dir=None,
                                 evo_candidates: int = 20, **kw):
    """
    The "approach K gate" pilot function. Identical to strategy_evolved_af
    in every respect (candidate generation, AF scoring via the sandbox) —
    the ONLY difference is the final batch-selection step: instead of pure
    top-k (select_batch), this uses the baseline's own novelty-weighted
    greedy selection (novelty_aware_select_vectorised, real defaults
    w_acq=0.9/w_nov=0.1 as used by strategy_mo_egbo_novelty itself), so
    the evolved AF's per-candidate scores feed a jointly-diverse batch
    instead of a top-k-by-score batch. Tests whether L's negative result
    was partly an artifact of the fixed top-k selection step rather than
    of per-candidate scoring quality itself.
    """
    try:
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement)
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import SumMarginalLogLikelihood
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.transforms import unnormalize
        from botorch.optim.optimize import optimize_acqf
        from pymoo.algorithms.moo.unsga3 import UNSGA3
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.core.termination import NoTermination
        from pymoo.util.ref_dirs import get_reference_directions
        from novelty_selection import novelty_aware_select_vectorised

        tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
        lo, hi = bounds[:, 0], bounds[:, 1]
        d = bounds.shape[0]
        M = Y_obs.shape[1]
        directions = oracle.objective_directions()

        Y_int = Y_obs.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_int[:, j] = -Y_int[:, j]

        X_norm = (X_obs - lo) / (hi - lo + 1e-12)
        train_x = torch.tensor(X_norm, **tkwargs)
        train_y = torch.tensor(Y_int, **tkwargs)
        standard_bounds = torch.zeros(2, d, **tkwargs)
        standard_bounds[1] = 1.0

        Y_all_int = oracle._Y_raw.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_all_int[:, j] = -Y_all_int[:, j]
        ref_point = torch.tensor(
            Y_all_int.min(axis=0) -
            0.1 * (Y_all_int.max(axis=0) - Y_all_int.min(axis=0) + 1e-9),
            **tkwargs)

        models = [SingleTaskGP(train_x, train_y[:, j:j + 1],
                                outcome_transform=Standardize(m=1))
                  for j in range(M)]
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll, max_attempts=1)

        acq_fn = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=ref_point, X_baseline=train_x,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
            prune_baseline=True, cache_root=True,
        )

        try:
            qbo_x, _ = optimize_acqf(
                acq_fn, bounds=standard_bounds, q=batch_size,
                num_restarts=2, raw_samples=16,
                options={"maxiter": 20},
            )
        except Exception:
            qbo_x = torch.rand(batch_size, d, **tkwargs)

        pf_idx_np = pareto_front_of(Y_obs, directions=directions)
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
            algo = UNSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=seed_x,
                          seed=int(rng.integers(1e6)))
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
                pad = np.tile(f_vals[-1:], (pop_size_actual - len(f_vals), 1))
                pop.set("F", np.vstack([f_vals, pad]))
            algo.tell(infills=pop)
            ea_x = torch.tensor(algo.ask().get("X"), **tkwargs)
        except Exception:
            ea_x = torch.rand(evo_candidates, d, **tkwargs)

        candidates = torch.cat([qbo_x, ea_x], dim=0)
        candidates_n = candidates.cpu().numpy()

        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            post = model.posterior(candidates)
            pool_mu = post.mean.detach().cpu().numpy()
            pool_sigma = post.variance.clamp_min(1e-12).sqrt().detach().cpu().numpy()

        front_allmax = to_allmax(Y_obs.copy(), directions=directions)
        ref_point_allmax = ref_point.cpu().numpy()
        step = len(X_obs)
        stagnant_batches = 0

        scores = run_af_in_sandbox(
            af_code, candidates_n, pool_mu, pool_sigma, X_norm,
            front_allmax, ref_point_allmax, step, budget, step, stagnant_batches,
            front_allmax,
            objective_names=oracle.objective_names(),
            log_dir=sandbox_log_dir,
        )
        # The only line that differs from strategy_evolved_af: joint
        # novelty-weighted selection instead of pure top-k, using the
        # SAME real defaults strategy_mo_egbo_novelty itself uses.
        selected_idx = novelty_aware_select_vectorised(
            candidates_n, np.asarray(scores, dtype=float), batch_size,
            w_acq=0.9, w_nov=0.1, X_obs_n=X_norm,
        )

        new_x_norm = candidates[selected_idx]
        new_x_raw = (unnormalize(new_x_norm, torch.tensor(
            np.column_stack([lo, hi]).T, **tkwargs)).cpu().numpy())

        return new_x_raw, {"evolved_af": True, "n_candidates": len(candidates),
                            "novelty_selection": True}

    except SandboxError as e:
        raise
    except Exception as e:
        warnings.warn(f"strategy_evolved_af_novelty candidate generation failed "
                       f"({e}), falling back to lightweight strategy_mo_egbo.")
        cands, extra = strategy_mo_egbo(oracle, X_obs, Y_obs, bounds, batch_size, rng)
        extra["evolved_af"] = False
        extra["fallback_reason"] = str(e)
        return cands, extra


def strategy_evolved_generation(oracle, X_obs, Y_obs, bounds, batch_size, rng,
                                 gen_code: str, budget: int, af_code: str = None,
                                 sandbox_log_dir=None, gen_sandbox_log_dir=None,
                                 evo_candidates: int = 20, **kw):
    """
    The "candidate-generation gate" pilot function — the third evolution
    surface (after scoring in strategy_evolved_af, selection in
    strategy_evolved_af_novelty). The GP-fit + scoring + top-k selection
    machinery is identical to strategy_evolved_af; what's REPLACED is the
    candidate pool itself: instead of qLogNEHVI-optimized + UNSGA3-evolved
    points, candidates come from a geometry-only sandboxed proposer (see
    gen_interface.py's contract — no GP/acquisition access, only X_obs,
    pareto_x, and campaign state). af_code defaults to SEED_PROGRAMS
    ["trust_only"] — the cleaner-null scorer from the composition gate —
    so this isolates the generation lever the same way strategy_evolved_af
    isolated scoring and strategy_evolved_af_novelty isolated selection:
    one axis varies, the rest of the pipeline stays fixed.

    No optimize_acqf/qLogNEHVI/UNSGA3 call at all in this path — the GP fit
    is still needed (to score whatever candidates the proposer returns),
    but candidate proposal itself no longer depends on botorch's or
    pymoo's own optimizers, which also makes this path meaningfully
    cheaper than strategy_evolved_af/strategy_evolved_af_novelty.
    """
    if af_code is None:
        af_code = SEED_PROGRAMS["trust_only"]
    try:
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import SumMarginalLogLikelihood
        from botorch.utils.transforms import unnormalize

        tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
        lo, hi = bounds[:, 0], bounds[:, 1]
        d = bounds.shape[0]
        M = Y_obs.shape[1]
        directions = oracle.objective_directions()

        Y_int = Y_obs.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_int[:, j] = -Y_int[:, j]

        X_norm = (X_obs - lo) / (hi - lo + 1e-12)
        train_x = torch.tensor(X_norm, **tkwargs)
        train_y = torch.tensor(Y_int, **tkwargs)

        Y_all_int = oracle._Y_raw.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_all_int[:, j] = -Y_all_int[:, j]
        ref_point = torch.tensor(
            Y_all_int.min(axis=0) -
            0.1 * (Y_all_int.max(axis=0) - Y_all_int.min(axis=0) + 1e-9),
            **tkwargs)

        models = [SingleTaskGP(train_x, train_y[:, j:j + 1],
                                outcome_transform=Standardize(m=1))
                  for j in range(M)]
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll, max_attempts=1)

        pf_idx_np = pareto_front_of(Y_obs, directions=directions)
        pareto_x_n = X_norm[pf_idx_np] if len(pf_idx_np) > 0 else np.zeros((0, d))

        step = len(X_obs)
        stagnant_batches = 0

        candidates_n = run_generator_in_sandbox(
            gen_code, X_norm, pareto_x_n, step, budget, step, stagnant_batches,
            d, evo_candidates, log_dir=gen_sandbox_log_dir,
        )
        candidates = torch.tensor(candidates_n, **tkwargs)

        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            post = model.posterior(candidates)
            pool_mu = post.mean.detach().cpu().numpy()
            pool_sigma = post.variance.clamp_min(1e-12).sqrt().detach().cpu().numpy()

        front_allmax = to_allmax(Y_obs.copy(), directions=directions)
        ref_point_allmax = ref_point.cpu().numpy()

        scores = run_af_in_sandbox(
            af_code, candidates_n, pool_mu, pool_sigma, X_norm,
            front_allmax, ref_point_allmax, step, budget, step, stagnant_batches,
            front_allmax,
            objective_names=oracle.objective_names(),
            log_dir=sandbox_log_dir,
        )
        selected_idx = select_batch(scores, batch_size)

        new_x_norm = candidates[selected_idx]
        new_x_raw = (unnormalize(new_x_norm, torch.tensor(
            np.column_stack([lo, hi]).T, **tkwargs)).cpu().numpy())

        return new_x_raw, {"evolved_generation": True, "n_candidates": len(candidates)}

    except (SandboxError, GenSandboxError) as e:
        raise
    except Exception as e:
        warnings.warn(f"strategy_evolved_generation candidate generation failed "
                       f"({e}), falling back to lightweight strategy_mo_egbo.")
        cands, extra = strategy_mo_egbo(oracle, X_obs, Y_obs, bounds, batch_size, rng)
        extra["evolved_generation"] = False
        extra["fallback_reason"] = str(e)
        return cands, extra


def run_2b_campaign(af_code: str, log: dict, batch_size: int = None,
                     seed: int = 0, sandbox_log_dir=None,
                     oracle_family: str = "excipient") -> dict:
    """
    Reconstruct the held-out campaign's oracle and initial points from its
    logged X_init/Y_init, then run a full sequential BO loop with the
    evolved AF as the acquisition function. Returns run_mo_campaign's
    result dict (hv_trajectory, X_obs, Y_obs, ...) plus final_hv.

    oracle_family is passed straight through to reconstruct_oracle — see
    its docstring. Defaults to "excipient" so every existing caller
    (evolve_af_2b.py, validate_2b_seed.py, validate_population_2b.py,
    run_2b_diagnostic.py) is unaffected.
    """
    torch.manual_seed(seed)  # see run_baseline_campaign's docstring note
    disc_oracle = reconstruct_oracle(log, oracle_family=oracle_family)
    X_init = np.array(log["X_init"])
    Y_init = np.array(log["Y_init"])
    budget = log["budget"]
    batch_size = batch_size or log["batch_size"]

    # Fresh per-campaign list, mutated in place by strategy_evolved_af
    # across every batch of THIS campaign (run_mo_campaign reuses this same
    # dict — and therefore this same list object — unchanged across its
    # per-batch loop, never recopying it) — see strategy_evolved_af's
    # docstring and the module docstring's "FIXED" note. Without this,
    # stagnant_batches would be hardcoded 0 for the whole campaign, same as
    # before the fix.
    strategy_kwargs = {"af_code": af_code, "budget": budget,
                        "sandbox_log_dir": sandbox_log_dir, "hv_history": []}
    result = run_mo_campaign(
        disc_oracle, X_init, Y_init, budget, strategy_evolved_af,
        strategy_kwargs,
        batch_size=batch_size, seed=seed,
    )
    result["final_hv"] = result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")
    return result


def run_baseline_campaign(log: dict, batch_size: int = None, seed: int = 0,
                           oracle_family: str = "excipient") -> dict:
    """
    Same reconstructed oracle/init points as run_2b_campaign, but running
    the REAL, unmodified strategy_mo_egbo_novelty — not an evolved AF. This
    is the missing reference point for Step B: without it, evolved AFs'
    2b final_hv numbers have no baseline to be judged competitive against.
    Uses the SAME held-out log (hence same oracle/init points) and the SAME
    seed convention as run_2b_campaign (seed=i per campaign index) so the
    comparison is apples-to-apples, not just similarly-generated.

    oracle_family: see reconstruct_oracle's docstring; must match whatever
    run_2b_campaign was called with for the same log, or the two aren't
    actually comparing the same oracle.

    torch.manual_seed(seed) here and in run_2b_campaign fixes a real gap:
    GP hyperparameter fitting, qLogNEHVI's MC sampler, and optimize_acqf's
    restarts all draw from torch's GLOBAL random state, which was never
    seeded — only the parts using the passed numpy `rng` were controlled.
    This is why the baseline's own mean HV came out different across
    separate script invocations on nominally "the same" 20 campaigns/seeds
    (7789.6, 8458.2, 8166.9 across three runs) — real run-to-run noise from
    unseeded torch internals, not a bug in the pairing logic itself
    (within a single run, af-vs-baseline comparisons were always fair,
    since both draw from the same process's evolving torch state
    sequentially). Absolute HV numbers reported anywhere going forward
    should come from a single run made with this fix in place, not mixed
    across pre-fix and post-fix invocations.
    """
    torch.manual_seed(seed)
    disc_oracle = reconstruct_oracle(log, oracle_family=oracle_family)
    X_init = np.array(log["X_init"])
    Y_init = np.array(log["Y_init"])
    budget = log["budget"]
    batch_size = batch_size or log["batch_size"]

    result = run_mo_campaign(
        disc_oracle, X_init, Y_init, budget, strategy_mo_egbo_novelty, {},
        batch_size=batch_size, seed=seed,
    )
    result["final_hv"] = result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")
    return result
