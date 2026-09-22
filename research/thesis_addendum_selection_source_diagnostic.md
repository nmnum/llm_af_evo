# Thesis addendum — selection-by-source diagnostic for §4.4

No editable thesis source (Word/LaTeX) exists in this repo — only `thesis-neha-final.pdf`.
The text below is written to slot directly into the existing document at the two
locations noted. Numbers are exact from `.scratch/check_llm_selection_source.py`
(raw per-batch data: `.scratch/llm_selection_source_diagnostic.json`).

---

## Insert at the end of §4.4 "Injection Architecture Determines the Wrong Prior's Effect"

(after the paragraph ending "...Injection persistence, not excipient chemistry, is
the mechanism behind any apparent wrong-prior advantage.")

> A related objection concerns the fairness of the pooled comparison itself, not
> just injection persistence. EGBO's own candidates are constructed via gradient
> ascent (`optimize_acqf`) and evolutionary search run directly against
> qLogNEHVI, while LLM candidates are proposed once from chemistry reasoning and
> merely scored, never refined against the acquisition surface. If GP candidates
> are systematically pre-optimised for the exact metric that decides the winner,
> `mo_llm_candidate_gen`'s null result (Table 4.3) could reflect this structural
> asymmetry rather than an absence of chemical value in the LLM's proposals.
>
> An exploratory selection-by-source diagnostic tests this directly: across 18
> batches (n = 3 seeds, mAb_aggregation, L1 prior; qwen2.5:14b-instruct
> substituted for the thesis's qwen3:32b for wall-clock reasons — qwen3:32b runs
> CPU-bound on the available hardware, so this check is indicative rather than
> confirmatory), LLM-sourced candidates made up 32.0% of the offered pool and won
> 25.6% of selections (23/90). Their mean acquisition value across batches was
> *higher* than EGBO's (−0.57 vs. −1.13), but EGBO's best candidate per batch
> exceeded the LLM's best more often (mean per-batch max 5.99 vs. 4.67), and in 5
> of 18 batches (28%) the LLM's entire proposal set won zero selections. LLM
> candidates are therefore competitive on average rather than structurally
> excluded, but modestly under-selected relative to their pool share (≈80% of a
> proportional share). The null result in Table 4.3 is not an artefact of the
> pooled comparison being rigged against the LLM outright, though the shutout
> rate indicates the acquisition-optimised GP side retains a real, if partial,
> structural edge that merit-scoring narrows without fully removing.

Optional companion table (Table 4.8, if a table is preferred over inline prose):

| | LLM-sourced | EGBO-sourced |
|---|---|---|
| Offered (pool share) | 32.0% (212) | 68.0% (450) |
| Selected (winners) | 25.6% (23) | 74.4% (67) |
| Mean acquisition value | −0.57 | −1.13 |
| Mean per-batch max acquisition value | 4.67 | 5.99 |
| Batches with zero LLM selections | 5/18 (28%) | — |

*Table 4.8. Selection-by-source diagnostic for* `mo_llm_candidate_gen`*, exploratory
(n = 3 seeds, mAb_aggregation, L1 prior, qwen2.5:14b-instruct).*

---

## Insert into §5.4 Limitations, model-scope paragraph

(append after "...may not replicate under models with weaker pharmaceutical
pretraining.")

> A related, narrower check (§4.4) used an even smaller substitute model
> (qwen2.5:14b-instruct) to test whether `mo_llm_candidate_gen`'s null result is
> an artefact of GP candidates being structurally pre-optimised against the
> scoring metric. That diagnostic is itself model-scope-limited and was not run
> with qwen3:32b for the same wall-clock reason.

---

## Insert into §5.5 Future work

(as a new paragraph, after the injection-persistence generalisation paragraph)

> A fifth line follows directly from the selection-by-source diagnostic (§4.4):
> confirming it at full statistical power with the thesis's own model
> (qwen3:32b, run unattended given its wall-clock cost) across both protein
> profiles and all three prior levels, rather than the single-cell, reduced-model
> exploratory check reported here. A natural extension beyond replication is
> testing whether the LLM's shutout rate reflects genuinely less informative
> chemistry or simply less local optimisation against the acquisition surface —
> e.g. by giving each LLM-proposed candidate a short local refinement step
> (gradient ascent or a small evolutionary nudge on `qLogNEHVI`) before pooling,
> then re-measuring whether the selection gap narrows.

---

## Reproducing / extending this check

`.scratch/check_llm_selection_source.py <n_seeds> [--mock] [--model=NAME]`

- Diagnostic fields (`n_llm_selected`, `n_egbo_selected`, `llm_acq_mean/max`,
  `egbo_acq_mean/max`) were added to `strategy_llm_candidate_gen.py`'s per-batch
  return dict — additive only, does not change candidate selection logic, so it
  is safe to rerun against any already-published `mo_llm_candidate_gen` config.
- To reproduce with the thesis's exact model: `python3 -u
  .scratch/check_llm_selection_source.py 25 --model=qwen3:32b`, matching Phase
  2's n=25 seeds; expect several hours given CPU-bound qwen3:32b latency
  (~577–595s per seed observed with the smaller substitute model, likely several
  times longer per seed at 32b — run detached/overnight).
- To extend across the full factorial: loop the script's `PROTEIN` /
  `PRIOR_LEVEL` constants over both profiles and all three prior levels to match
  Table 4.3's full design.
