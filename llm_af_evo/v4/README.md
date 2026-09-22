# v4 — exposing EGBO's real scoring signal to evolved AFs

Started from a direct question in the af-evolution branch conversation:
"what if we gave EGBO as base seed?" A faithful port turned out to be
impossible under v3's contract — investigated directly (read
`strategy_ls_na_egbo.py`/`novelty_selection.py`, not assumed) — for two
reasons:

1. EGBO's real per-candidate score is botorch's own qLogNEHVI acquisition
   value (a Monte-Carlo integral over the FULL joint objective posterior),
   not GP mean/std. `score_pool`'s `context` never exposed that value.
2. EGBO's selection is a SEQUENTIAL greedy loop (`novelty_aware_select_
   vectorised`): each pick's novelty term is computed against points
   already selected earlier in the SAME batch, which a one-shot,
   order-independent per-candidate score list cannot express.

(2) is a genuine, unresolved limitation of the `score_pool` contract —
not fixed in v4. (1) turned out to be fixable, and cheaply: EGBO's
`acq_fn` was already being built and called by `strategy_evolved_af` for
`optimize_acqf`, just never scored against the merged candidate pool. One
extra batched `acq_fn(candidates)` call exposes it.

## What's here

- `llm_af_evo/shared/full_replay.py` — `strategy_evolved_af` now also
  computes `acq_vals = acq_fn(candidates.unsqueeze(1))` (same call
  `strategy_mo_egbo_novelty` itself makes) and passes it to
  `run_af_in_sandbox` as `pool_acq_value`.
- `llm_af_evo/shared/sandbox.py` — `pool_acq_value` threaded through as a
  new, optional, additive `argv` slot (same established pattern as
  `obj_correlation`/`front_boundary_std`/`front_allmax_init`), min-max
  normalised into `context["pool"][i]["acq_value_norm"]` (same
  normalisation EGBO's own selection uses). Defaults to a flat,
  uninformative 0.5 for any caller that doesn't pass it — confirmed via
  direct unit tests that (a) `trust_only`'s `final_hv` on a real campaign
  is byte-identical before/after this change, (b) any AF that ignores
  `acq_value_norm` is completely unaffected whether or not the caller
  passes `pool_acq_value`.
- `src/af_interface_v4.py` — adds `egbo_novelty_like` to `SEED_PROGRAMS`:
  `0.9*acq_value_norm + 0.1*novelty-to-observed`, matching
  `strategy_mo_egbo_novelty`'s real `w_acq`/`w_nov` defaults exactly (the
  closest static approximation of EGBO's actual scoring achievable in
  this contract — it only omits EGBO's sequential within-batch term).
  Also adds `acq_value_progress_blend` to `STRATEGY_HINTS`, explicitly
  warning the LLM's gen-0 bootstrap against re-deriving hypervolume
  improvement by hand now that the real value is available directly.
- `src/evolve_af_v4.py` — copy of `evolve_af_v3.py`; `_build_system_prompt`
  updated to document `acq_value_norm` and rewrite the "baseline you're
  trying to beat" section to point the LLM at using it directly instead
  of approximating it. Everything else (fitness metric, bootstrap CI,
  MECHANISM_FAMILIES/champion-rehash anti-mode-collapse guards,
  checkpoint/resume) is unchanged from v3.
- `experiments/run_2b_diagnostic_v4.py` — copy of `run_2b_diagnostic_v3.py`,
  pointed at `af_interface_v4`/`egbo_novelty_like`; defaults `--heldout_dir`
  to v3's heldout set (reused directly, not duplicated — see below).

v4 has no `experiments/training_logs_tunable/` of its own: the training
logs only store oracle/init state (`X_init`/`Y_init`/`oracle_Y_raw`/etc.),
which `full_replay.py` refits and re-scores fresh on every campaign
replay regardless of `acq_value_norm` — nothing about the log format
needed to change, so v4 points `--train_dir`/`--heldout_dir` at v3's
existing 48-train/16-heldout set directly.

## Validation before building any of this out

Direct measurement, not assumption, on one real heldout campaign
(`run_2b_campaign`, oracle_family="tunable"):

| AF | final_hv |
|---|---|
| `trust_only` (unaffected by this change — regression check) | 10.634 (byte-identical to its pre-change value) |
| pure `acq_value_norm` (no novelty term at all) | 10.922 |
| `egbo_novelty_like` (0.9/0.1 real weights) | 10.901 |
| real EGBO-novelty baseline (this campaign) | ~10.86–10.94 range |

A **trivial** AF that just returns `acq_value_norm` directly already
lands inside the real baseline's range — closer than any hand-rolled
hypervolume-improvement approximation the LLM produced across 128+ calls
in v3's run1+run2 combined (champion margin there: +0.47–0.49%). This is
the validation that justified building the rest of v4 out, per the
"don't build on unvalidated mechanisms" standard this project has
followed throughout.

Confirmed via a mock-mode end-to-end smoke test: gen 0's `egbo_novelty_like`
seed alone reaches mean_margin=+0.65%, win_rate=0.75 on 4 training
campaigns — already ahead of v3's best evolved champion, before any
evolution happens.

## Open question, not solved here

EGBO's sequential within-batch novelty term (2, above) is still
unavailable to `score_pool`. If evolution under v4 plateaus meaningfully
below the real baseline specifically because of within-batch redundancy
(candidates clustering rather than spreading across a batch), the next
step would be a genuinely bigger contract change — e.g. an alternative
`select_batch(context) -> list[int]` entrypoint giving an evolved AF full
control over sequential selection, not just per-candidate scoring — not
attempted here since (1) hadn't even been validated as the binding
constraint before that effort would be worth spending.

## Status: run1 exists, not yet written up here

A real `evolve_af_v4.py` run has since been done:
`experiments/evolution_runs/run1/` (144 generations, real LLM) —
champion (`run1/best_af.py`) is `acq_value_norm` plus a small fixed-weight
(0.1116) GP-std bonus, i.e. evolution converged on almost exactly the
`egbo_novelty_like` seed shape this README's validation table already
flagged as close to the real baseline. `checkpoint.json`'s
`best_fitness_ever` (~0.00032) is in the same range as v3's champions.
There's also a `diagnostic_v4/` directory (48 logged AF calls, no
checkpoint/history — looks like a `run_2b_diagnostic_v4.py` diagnostic
pass, not a full evolution run). Neither has been analyzed or written up
in this document yet — treat the numbers above as a pointer for whoever
picks this up next, not a conclusion.
