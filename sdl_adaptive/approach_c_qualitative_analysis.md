# Qualitative Analysis: Approach C Code Logs

**Source:** `sdl_adaptive/approach_c_code_logs/call_0000.py`–`call_0017.py` (18 files, 1351 lines total)

Approach C asks an LLM to rewrite the entire `suggest()` optimiser function as executable Python each call, run in a subprocess sandbox against `optimiser_template.py`. `approach_c_evo` is the variant where the generated code additionally has access to `evolutionary_candidates()` / `novelty_select()` helpers. Both conditions are logged into this same directory.

## Status

- **Done:** qualitative read of all 18 code logs; identified the dead-acquisition-score bug (below).
- **Done:** bug-fix proxy rerun (deterministic replay of the LLM's own patched Generation-C code, no live LLM calls, 80 seeds matching `results_power2`'s protocol) — AUC 0.708 → 0.793.
- **Done:** live rerun of `approach_c_evo` (real `qwen3-coder:30b` calls) under the same 20-seed × 5-dataset protocol as `approach_a`/`b`/`c`/`d` in `results_combined`, giving `approach_c_evo` a properly comparable figure for the first time — see "Comparison across all conditions" below.

## Three generations, not eighteen

Despite 18 files, the LLM produced only **three structurally distinct implementations**; most calls are exact or near-exact repeats.

**Generation A — `call_0000`, `call_0001`** (2/18, 50 lines)
Plain GP-UCB: fit a `GaussianProcessRegressor(Matern(nu=2.5), normalize_y=True, n_restarts_optimizer=3)`, sample 300 uniform-random candidates, score with `mu + 2.0*sigma`, take argmax. No evolutionary-candidate import — this is the base `approach_c` behaviour. `call_0000.py` uniquely carries inline `[cite: N]` comment annotations (e.g. `# Instantiate the model correctly as an object, not a class alias[cite: 12]`); `call_0001.py` is the identical code with the annotations stripped — likely an artifact of the LLM citing retrieved system-prompt context on its first call, not a functional difference.

**Generation B — `call_0002` through `call_0013`** (12/18, 77–78 lines)
Adds the evolutionary/novelty machinery (`approach_c_evo`). Runs the Generation-A GP-UCB block first, then, if `n > d + 5`, **re-generates a second, independent 300-point candidate pool**, re-predicts on it with the already-fitted model, merges it with 72 evolutionary candidates, and calls `novelty_select(..., merit_weight=0.7)`. The first GP-UCB computation's result is discarded whenever this second branch fires — a real, repeated inefficiency (duplicated model-predict calls). All 12 files are structurally identical to each other; `call_0004` and `call_0006`–`call_0013` drop an unused `from scipy.stats import norm` import present in `call_0002`/`call_0003`, otherwise the code is byte-identical.

**Generation C — `call_0014` through `call_0017`** (4/18, 81–82 lines)
A genuine refactor: collapses Generation B's two sequential blocks into a single `if n > d + 5: ... elif n > d: ...`, so the GP is fit once and the candidate pool generated once per call, eliminating Generation B's redundant double-fit/double-predict. Functionally equivalent otherwise.

## The substantive bug: dead acquisition scores for evolutionary candidates

In every evo-capable version (Generations B and C, 16/18 files), evolutionary candidates are merged into the scoring pool with their acquisition score hardcoded to zero:

```python
acq_scores = np.concatenate([gp_scores, np.zeros(72)])  # dummy scores for evo candidates
```

`novelty_select(..., merit_weight=0.7)` combines `0.7*merit + 0.3*novelty`. Since merit is unconditionally 0 for every evolutionary candidate, they can only ever be selected on the strength of the 0.3-weighted novelty term — structurally starved relative to GP candidates regardless of how promising the evolutionary points actually are. The comment ("dummy scores") shows the LLM knew this was a stand-in, and it was never fixed across 16 subsequent identical or near-identical calls.

This lines up with the quantitative finding originally recorded in `sdl_adaptive/COMPREHENSIVE_LOG.md` (`results_power2`): `approach_c_evo` looked like the weakest LLM-adjacent condition sampled there (0.708 AUC vs. `egbo`'s 0.805 and `novelty_egbo`'s 0.791; separately, on `pareto_20210112`, `novelty_egbo` reaches 0.833 and `egbo` 0.818). This code log gives a plausible causal mechanism for that gap — the LLM wired up the evolutionary-candidate pool but never connected it to real acquisition scoring, so the "diversity" feature it added was largely cosmetic.

> **Protocol caveat (historical).** The 0.708 figure came from `results_power2`: **80 seeds on a single dataset** (`pareto_20210112`), which is not directly comparable to the current `approach_a`/`b`/`c`/`d` numbers in `results_combined` (**20 seeds × all 5 datasets**). This has since been resolved — see "Comparison across all conditions" below for `approach_c_evo`'s properly comparable figure.

**Confirmed by a bug-fix proxy rerun.** To test this causally rather than just narratively, I took the LLM's own Generation-C code (`call_0017.py`) and patched only the one line — `evo_cands` now score with the same `mu + beta*sigma` GP prediction the plain candidates get, instead of a hardcoded zero — then replayed it deterministically (no further LLM calls) across the identical 80-seed protocol used for the original `results_power2` figure (`pareto_20210112`, `n_init=5`, `controller_interval=5`, same `generate_init_points()` draw per seed). Result: **AUC rose from 0.708 to 0.793 (n=80, std 0.117)** — closing roughly 88% of the gap to `egbo` (0.805) and landing essentially level with `novelty_egbo` (0.791). This confirms the dead-acquisition-score bug was not an incidental style issue but a substantial share of the measured performance gap: fixing this one line, with everything else about the LLM's code held fixed, recovers most of what separates `approach_c_evo` from the hand-designed EGBO baselines on this one dataset. (Results: `sdl_adaptive/results_fixed_evo_proxy/pareto_20210112/approach_c_evo_fixed/`.)

## Comparison across all conditions (live rerun, 20 seeds × 5 datasets)

The bug-fix proxy above is a frozen replay of one saved code snapshot on one dataset. To get a real, comparable figure, `approach_c_evo` was rerun **live** (actual `qwen3-coder:30b` calls, fresh code generated per seed) under the exact protocol used for `approach_a`/`b`/`c`/`d` in `results_combined`: 20 seeds × all 5 datasets, `n_init=5`, `controller_interval=5`. This is the first time `approach_c_evo` has a figure directly comparable to the other LLM conditions.

| dataset | approach_c_evo | best other LLM approach | egbo | novelty_egbo |
|---|---|---|---|---|
| pareto_20210112 | **0.905** | approach_a 0.820 | 0.892 | 0.908 |
| pareto_20201218 | 0.851 | approach_a 0.861 | 0.850 | 0.849 |
| coatings | 0.905 | approach_d 0.937 | 0.972 | 0.961 |
| pareto_20201223 | 0.853 | approach_c 0.869 | 0.923 | 0.917 |
| pareto_20210104 | **0.846** | approach_b 0.788 | 0.860 | 0.872 |

**Overall mean AUC across all 5 datasets:**

| condition | mean AUC |
|---|---|
| novelty_egbo | 0.901 |
| egbo | 0.899 |
| **approach_c_evo** | **0.851** |
| approach_a | 0.842 |
| approach_c | 0.840 |
| approach_b | 0.833 |
| approach_d | 0.818 |

This overturns the earlier reading. Under the old single-dataset, 80-seed sample, `approach_c_evo` looked like the weakest LLM condition (0.708). Under the fair, broader protocol, it is the **strongest LLM-authored condition on average** (0.851), ahead of `approach_a`, `approach_c`, `approach_b`, and `approach_d` — and on two of five datasets (`pareto_20210112`, `pareto_20210104`) it matches or beats both hand-designed EGBO baselines outright. The remaining gap to EGBO overall (~0.05) is real but much smaller and more consistent than the original 0.708 figure implied, and it is not uniform: `approach_c_evo` underperforms most on `coatings`, where EGBO's margin is largest across the board.

The dead-acquisition-score bug documented above is a real, self-inflicted regression in the *specific saved code snapshots* this analysis inspected (Generations B/C, 16/18 logged calls) — but it evidently isn't present or isn't dominant in every live generation the LLM produces, since the live rerun's fresh-per-seed code does not reproduce the old 0.708 result. The two findings aren't in tension: the bug-fix proxy shows that *when* the bug is present, fixing it recovers most of the gap; the live rerun shows the LLM does not reliably reproduce that exact bug across independent generations, and unconditionally, `approach_c_evo` performs competitively.

## Other qualitative observations

- **Fixed exploration weight, no decay.** `beta = 2.0` is hardcoded in all 18 versions — no progress-dependent decay schedule, unlike the hand-designed EGBO baselines elsewhere in the project. The LLM never explored this axis across any of its rewrites.
- **Random-uniform, not quasi-random, candidate sampling.** All versions draw candidates via `np.random.uniform` per dimension rather than Sobol/LHS, giving noisier space coverage than the hand-written baselines.
- **Defensive but opaque error handling.** Every version wraps its GP logic in a bare `try/except Exception: pass`, falling back to a pre-drawn random point. Good defensive practice — the function never crashes or returns `None` — but it silently swallows all exceptions, including genuine bugs, so a silent degradation to random search is indistinguishable from a healthy run without inspecting logs.
- **Near-total code stagnation.** 12 of 18 calls are exact duplicates of Generation B, and the LLM only ever iterated once further, to Generation C. For a design where the optimiser is rewritten from scratch every call, there is remarkably little exploration of alternative mechanisms — three structural variants across 18 independent generation calls.

## Bottom line

This log set is useful primary evidence for the thesis's "LLM-authored optimiser code underperforms hand-designed EGBO" claim, but the full picture is more nuanced than the code logs alone suggest. The 18 saved code snapshots show a real, self-inflicted integration bug (dead acquisition scores on evolutionary candidates), repeated verbatim across the majority of the LLM's rewrites without ever being caught — and a bug-fix proxy rerun confirms that, when that bug is present, fixing that single line recovers most (≈88%) of the AUC gap to hand-designed EGBO on `pareto_20210112` (0.708 → 0.793, vs. `egbo`'s 0.805).

But the properly comparable live rerun (20 seeds × 5 datasets, matching `approach_a`/`b`/`c`/`d`'s protocol) tells a broader story: `approach_c_evo` is not, on average, the weakest LLM condition — it's the **strongest** one (mean AUC 0.851, ahead of `approach_a` 0.842, `approach_c` 0.840, `approach_b` 0.833, `approach_d` 0.818), and on two of five datasets it matches or beats both EGBO baselines outright. The 0.708 figure that motivated this whole investigation turns out to have been an artifact of a single unlucky dataset/protocol combination, not a representative measure of the condition. The residual gap to EGBO overall (~0.05) is real, and larger on some datasets (`coatings`) than others, but it's a much smaller and more qualified claim than "the LLM's optimiser code is broken and underperforms."

The interesting failure mode, then, isn't that the LLM's approach is fundamentally weaker — it's two separate things layered on top of each other: (1) the LLM can and does ship silent, self-introduced regressions in its generated code without catching them across repeated calls, evidenced directly in the 18 saved logs; and (2) despite that, averaged across a broader and fairer sample, its live-generated code is competitive with — and in some cases matches — hand-designed EGBO. Both are true, and both are worth reporting: the bug-fix proxy result is a clean within-sample causal demonstration of (1); the 5-dataset live rerun is the actual comparable headline number for (2).
