"""
da_coreg.py — the DA-COREG lever: a coregionalized multi-task GP surrogate
(shared cross-objective structure) as a drop-in alternative to the
independent per-objective SingleTaskGP/ModelListGP used everywhere else
in this project. Standard botorch tooling (MultiTaskGP, an ICM/Kronecker
coregionalization kernel) implements exactly this concept — objectives are
modelled as "tasks" that share a learned covariance, so observing one
objective informs the posterior on a correlated one, which independent
per-objective GPs cannot do by construction.

This is the ONLY place DA-COREG enters the pipeline: fit_da_coreg_model
returns an object exposing the same .posterior(candidates) -> (mean,
variance) interface that ModelListGP.posterior already provides, so every
downstream caller (strategy_evolved_af, strategy_compose_batch, etc.) is
unmodified by which surrogate was used — the swap is invisible past this
module's boundary, which is what makes the 2x2 ablation's cells
attributable to one lever at a time.

Not yet exercised against a real botorch install in this environment
(no numpy/torch/botorch available in this sandbox) — the MultiTaskGP
construction/posterior call signature below follows the documented API,
but is the single most likely place to need a version-specific tweak.
Flagging explicitly rather than asserting confidence: run
debug_da_coreg.py first and report the exact error if fit_da_coreg_model
raises, rather than assuming this is correct.
"""

import warnings

import numpy as np
import torch


class _DACoregPosterior:
    """Minimal wrapper matching the .mean/.variance interface botorch's
    own posterior objects expose, so callers written against
    ModelListGP.posterior(...)'s output shape need no changes."""

    def __init__(self, mean: torch.Tensor, variance: torch.Tensor):
        self.mean = mean
        self.variance = variance


class DACoregModel:
    """
    Wraps a fitted botorch MultiTaskGP so calling code can do
    model.posterior(candidates) exactly as it already does for ModelListGP,
    getting back (N, M) mean/variance across all M objectives jointly
    modelled by the coregionalized kernel — instead of M independent
    per-objective posteriors.

    task_mean/task_std: (M,) tensors holding each task's own train-set
    mean/std, used to undo the per-task standardization fit_da_coreg_model
    applies before fitting (see that function's docstring for why — a
    single pooled Standardize(m=1) over all tasks stacked together badly
    mis-scales whichever task's distribution differs most from the others,
    which on mAb caused the coregionalized kernel to fit a degenerate,
    near-zero predictive variance for 2 of 3 objectives). The raw model's
    posterior lives in per-task-standardized space; posterior() below maps
    it back to each task's original scale before returning.
    """

    def __init__(self, mt_model, n_tasks: int, tkwargs: dict,
                 task_mean: torch.Tensor, task_std: torch.Tensor):
        self._model = mt_model
        self._n_tasks = n_tasks
        self._tkwargs = tkwargs
        self._task_mean = task_mean
        self._task_std = task_std

    def posterior(self, candidates: torch.Tensor):
        # MultiTaskGP.posterior returns predictions for every task when
        # output_indices is left as the model's configured output_tasks
        # (set at construction in fit_da_coreg_model) — no per-call task
        # index needs to be appended to `candidates` here. Column order
        # matches task index 0..M-1, same order as task_mean/task_std.
        post = self._model.posterior(candidates)
        mean = post.mean * self._task_std + self._task_mean
        variance = post.variance * self._task_std ** 2
        return _DACoregPosterior(mean=mean, variance=variance)


def fit_da_coreg_model(train_x: torch.Tensor, train_y: torch.Tensor,
                        tkwargs: dict) -> DACoregModel:
    """
    Fit a single coregionalized multi-task GP jointly over all M columns
    of train_y (already in all-maximise convention, same as every other
    caller's train_y), sharing structure across objectives via an ICM
    task kernel instead of fitting M independent SingleTaskGPs.

    train_x: (n, d), train_y: (n, M) — same shapes strategy_evolved_af's
    ModelListGP construction takes.

    Standardization: each task column is z-scored using ITS OWN train
    mean/std before being stacked into the flat (n*M, 1) target tensor
    MultiTaskGP expects — NOT a single Standardize(m=1) applied to the
    pooled/flattened union of all M tasks' values. That pooled approach
    (the original implementation here) computes one global mean/std
    across all tasks combined; on mAb, where Tm/kD/viscosity have very
    different raw scales (Tm std~2, kD std~11.5, viscosity std~4, all
    around different means), the tasks that deviate most from the pooled
    mean/std get badly mis-scaled going into the shared ICM task-
    covariance fit. Diagnostic sweep (32 fits across 4 aggregation_
    tendency levels) confirmed this: Tm (the task nearest the pooled
    mean) fit reliably in every split, while kD/viscosity intermittently
    (4-7 of 8 splits per level) produced a degenerate near-zero
    predictive variance on held-out points (min_var as low as 1e-10,
    predictive NLL up to 1e10) — the exact asymmetry per-task
    standardization removes. DACoregModel.posterior() undoes this
    z-scoring per task before returning, so callers see predictions on
    the original Y scale exactly as before.
    """
    from botorch.models.multitask import MultiTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll

    n, d = train_x.shape
    M = train_y.shape[1]

    task_mean = train_y.mean(dim=0)
    task_std = train_y.std(dim=0).clamp_min(1e-6)
    train_y_std = (train_y - task_mean) / task_std

    # Stack (x, task_index) rows across all M objectives — MultiTaskGP's
    # expected input format: one flat (n*M, d+1) tensor with the task
    # index in the last column (task_feature=-1 below), and a matching
    # flat (n*M, 1) target tensor, now per-task-standardized.
    task_col_blocks = []
    y_blocks = []
    for m in range(M):
        task_col = torch.full((n, 1), float(m), **tkwargs)
        task_col_blocks.append(torch.cat([train_x, task_col], dim=-1))
        y_blocks.append(train_y_std[:, m:m + 1])
    train_x_aug = torch.cat(task_col_blocks, dim=0)
    train_y_aug = torch.cat(y_blocks, dim=0)

    # No outcome_transform here — standardization is already done above,
    # per task, before this point; an additional Standardize(m=1) would
    # just re-introduce the pooled-across-tasks mis-scaling this fix
    # removes.
    model = MultiTaskGP(
        train_x_aug, train_y_aug, task_feature=-1,
        output_tasks=list(range(M)),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(mll, max_attempts=1)

    return DACoregModel(model, n_tasks=M, tkwargs=tkwargs,
                         task_mean=task_mean, task_std=task_std)
