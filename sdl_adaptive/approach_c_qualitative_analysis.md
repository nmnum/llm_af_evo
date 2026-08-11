# Qualitative Analysis: Approach C Code Logs

**Source:** `sdl_adaptive/approach_c_code_logs/call_0000.py`–`call_0017.py` (18 files, 1351 lines total)

Approach C asks an LLM to rewrite the entire `suggest()` optimiser function as executable Python each call, run in a subprocess sandbox against `optimiser_template.py`. `approach_c_evo` is the variant where the generated code additionally has access to `evolutionary_candidates()` / `novelty_select()` helpers. Both conditions are logged into this same directory.

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

This lines up with the quantitative finding already recorded in `sdl_adaptive/COMPREHENSIVE_LOG.md` (`results_power2`): `approach_c_evo` was the weakest LLM-adjacent condition sampled (0.708 AUC vs. `egbo`'s 0.805 and `novelty_egbo`'s 0.791; separately, on `pareto_20210112`, `novelty_egbo` reaches 0.833 and `egbo` 0.818). This code log gives a plausible causal mechanism for that gap — the LLM wired up the evolutionary-candidate pool but never connected it to real acquisition scoring, so the "diversity" feature it added was largely cosmetic.

**Confirmed by a bug-fix proxy rerun.** To test this causally rather than just narratively, I took the LLM's own Generation-C code (`call_0017.py`) and patched only the one line — `evo_cands` now score with the same `mu + beta*sigma` GP prediction the plain candidates get, instead of a hardcoded zero — then replayed it deterministically (no further LLM calls) across the identical 80-seed protocol used for the original `results_power2` figure (`pareto_20210112`, `n_init=5`, `controller_interval=5`, same `generate_init_points()` draw per seed). Result: **AUC rose from 0.708 to 0.793 (n=80, std 0.117)** — closing roughly 88% of the gap to `egbo` (0.805) and landing essentially level with `novelty_egbo` (0.791). This confirms the dead-acquisition-score bug was not an incidental style issue but a substantial share of the measured performance gap: fixing this one line, with everything else about the LLM's code held fixed, recovers most of what separates `approach_c_evo` from the hand-designed EGBO baselines. (Results: `sdl_adaptive/results_fixed_evo_proxy/pareto_20210112/approach_c_evo_fixed/`.)

## Other qualitative observations

- **Fixed exploration weight, no decay.** `beta = 2.0` is hardcoded in all 18 versions — no progress-dependent decay schedule, unlike the hand-designed EGBO baselines elsewhere in the project. The LLM never explored this axis across any of its rewrites.
- **Random-uniform, not quasi-random, candidate sampling.** All versions draw candidates via `np.random.uniform` per dimension rather than Sobol/LHS, giving noisier space coverage than the hand-written baselines.
- **Defensive but opaque error handling.** Every version wraps its GP logic in a bare `try/except Exception: pass`, falling back to a pre-drawn random point. Good defensive practice — the function never crashes or returns `None` — but it silently swallows all exceptions, including genuine bugs, so a silent degradation to random search is indistinguishable from a healthy run without inspecting logs.
- **Near-total code stagnation.** 12 of 18 calls are exact duplicates of Generation B, and the LLM only ever iterated once further, to Generation C. For a design where the optimiser is rewritten from scratch every call, there is remarkably little exploration of alternative mechanisms — three structural variants across 18 independent generation calls.

## Bottom line

This log set is useful primary evidence for the thesis's "LLM-authored optimiser code underperforms hand-designed EGBO" claim, and — unlike the aggregate AUC tables alone — supplies a causal explanation, now empirically confirmed rather than just plausible: the LLM introduced a real integration bug (dead acquisition scores on its own evolutionary candidates), repeated it verbatim across the majority of its rewrites without noticing, and fixing that single line recovers most (≈88%) of the AUC gap to hand-designed EGBO. The residual gap (0.793 vs. `egbo`'s 0.805) is small enough to be within noise (seed SDs of 0.12–0.16 on this dataset) — so once this one bug is corrected, the LLM's own code is roughly competitive with the hand-tuned baseline on this dataset. The interesting failure mode, then, isn't that the LLM's approach is fundamentally weaker — it's that the LLM shipped a silent, self-introduced regression and never caught it across 16 further calls.
