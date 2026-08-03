# LS-NA-EGBO: LLM-Seeded Novelty-Aware EGBO

Code for benchmarking LLM-augmented optimization against EGBO for multi-objective
biologic formulation optimization (max Tm, max kD, min viscosity).

## Quick Start

```bash
# Install dependencies
uv pip install botorch pymoo torch scikit-learn matplotlib seaborn

# Pull the LLM (required for any non-mock run; qwen3:32b, not qwen2.5:72b-instruct)
ollama pull qwen3:32b

# Phase 1: reduced space, mock LLM (no Ollama needed)
python run_benchmark_resumable.py --phase 1 --mock_llm --n_seeds 15 \
  --budget 30 --n_init 10 --batch_size 5 \
  --proteins mAb_aggregation mAb_oxidation \
  --priors L1 blank --w_nov 0.1 \
  --out_dir ./results/benchmark_phase1

# Phase 2: full 14-excipient space, real LLM (requires Ollama)
# Model: qwen3:32b (switched from qwen2.5:72b-instruct — 1.7x faster on this
# hardware, 327s vs 555s per warm-start call, no parsing issues; see PLAN.md
# "Model selection" for the measured comparison). Conditions restricted to
# the 5 core ones for this pass; mo_ls_egbo/mo_llm_candidate_gen deferred separately.
python run_benchmark_resumable.py --phase 2 --n_seeds 25 \
  --budget 40 --n_init 10 --batch_size 5 \
  --model qwen3:32b \
  --proteins mAb_aggregation mAb_oxidation \
  --priors L1 blank wrong --w_nov 0.1 \
  --conditions mo_random mo_egbo mo_egbo_real mo_egbo_novelty mo_ls_na_egbo \
  --out_dir ./results/benchmark_phase2

# Sensitivity sweep (novelty weight tuning)
python run_sensitivity.py --mock_llm --n_seeds 15 --budget 30 \
  --proteins mAb_aggregation mAb_oxidation \
  --priors L1 blank \
  --out_dir ./results/benchmark_sensitivity

# Generate figures
python generate_all_figures.py
```

## File Structure

### Foundation (existing code, preserved)
- `excipient_oracle.py` — Excipient catalogue, dose-response curves, 16D encoding
- `excipient_oracle_mo.py` — Multi-objective oracle (Tm, kD, viscosity)
- `excipient_campaign_mo.py` — Campaign runner + baseline strategies (random, EGBO, EGBO-real, existing LLM)

### New strategies
- `novelty_selection.py` — Aqeeli et al. novelty-aware batch selection (default w_acq=0.9, w_nov=0.1; 0.3 is the Aqeeli et al. default, found too aggressive at budget=30 in the original sweep — see PLAN.md for caveats on that finding after the prompt rewrite)
- `llm_warmstart.py` — LLM warm-start with 3-layer prompt + mock mode + Kennard-Stone diversity. Acquisition scoring in the EGBO stage (`strategy_ls_na_egbo.py`) is vectorised (single batched call, not a per-candidate loop).
- `strategy_ls_na_egbo.py` — Recommended: LLM warm-start + novelty-aware EGBO
- `strategy_llm_candidate_gen.py` — LLM-as-candidate-generator: LLM generates candidates every batch, merged into EGBO's acquisition loop. NOTE: originally labeled "LABO" but does not implement the actual LABO paper's multi-fidelity gating mechanism — see the file's docstring for details.

### Runners
- `run_benchmark_resumable.py` — Main runner (Phases 1 & 2), checkpointed
- `run_sensitivity.py` — Novelty weight sweep + isolated warm-start condition
- `generate_all_figures.py` — 6 figure types

### Phase 3 (coatings generalisability)
- `synthetic_coatings_oracle.py` — 4D continuous coatings oracle
- `run_phase3.py` — Phase 3 benchmark runner

### Legacy
- `run_benchmark.py` — Original runner (superseded by _resumable)
- `generate_figures.py` — Original figure generator (superseded by generate_all_figures)
- `excipient_adapter.py` — Adapter for existing code
- `posthoc_llm_comparison.py` — LLM-BO vs EGBO on Ada coatings

## Key Findings (mock LLM, provisional)

1. **w_nov=0.1 is the sweet spot** (not 0.3 from Aqeeli et al.) — 0.3 is too aggressive at budget=30
2. **Mock LLM warm-start helps sample efficiency** (saves 2-6 experiments to 70% HV) but **hurts final HV** (-2.5% to -12.8%) because it narrows the initial design space
3. **mo_egbo_real (qLogNEHVI + U-NSGA-III, no novelty) is the strongest baseline** at budget=30
4. **Real 72B LLM runs are the decisive test** — mock LLM is not a valid proxy

## Architecture

```
Stage 1: LLM Warm-Start (once, before experiments)
  → 3-layer prompt (reasoning / domain physics / materials)
  → Generate n_propose candidates
  → Kennard-Stone diversity selection → n_init initial points

Stage 2: Novelty-Aware EGBO (autonomous)
  → Per-objective GPs (Matern 5/2)
  → qLogNEHVI acquisition (BoTorch) + U-NSGA-III evolutionary candidates (pymoo)
  → Novelty-aware batch selection: score = 0.9*acq + 0.1*novelty
  → Batch size = 5
```

## Conditions Benchmarked

| Condition | Init | Acquisition | Purpose |
|-----------|------|-------------|---------|
| mo_random | Random | Random | Floor baseline |
| mo_egbo | Random | Lightweight GP+Pareto | Lightweight EGBO |
| mo_egbo_real | Random | qLogNEHVI+U-NSGA-III | Real EGBO (strongest baseline) |
| mo_egbo_novelty | Random | qLogNEHVI+U-NSGA-III+novelty | Aqeeli et al. replication |
| mo_ls_egbo | LLM | qLogNEHVI+U-NSGA-III (no novelty) | Isolated warm-start effect |
| mo_ls_na_egbo | LLM | qLogNEHVI+U-NSGA-III+novelty | **Recommended architecture** |
| mo_llm_candidate_gen | Random | LLM every batch + qLogNEHVI scoring | LLM-as-candidate-generator (not literature LABO) |
