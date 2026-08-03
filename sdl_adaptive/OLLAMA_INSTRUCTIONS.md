# Running Real LLM Conditions with Ollama

This guide covers running the three real LLM controllers (Approach A, B, C) locally
using Ollama, and merging those results back into the existing mock-controller benchmark.

---

## Prerequisites

### 1. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download installer from https://ollama.com/download
```

### 2. Pull the model

```bash
ollama pull qwen2.5-coder:7b
```

Verify it works:
```bash
ollama run qwen2.5-coder:7b "Reply with: OK"
```

### 3. Install Python dependencies

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn ollama requests
```

---

## Directory layout expected

```
sdl_adaptive/
├── data/                        ← ADA datasets (download first)
│   ├── coatings_2022.csv
│   ├── campaign 2020-12-18_17-38-40.csv
│   ├── campaign 2020-12-23_17-06-50.csv
│   ├── campaign 2021-01-04_08-37-39.csv
│   └── campaign 2021-01-12_16-26-56.csv
├── results/                     ← existing mock results (from reproduce.ipynb)
│   └── metrics_summary.csv
├── oracle.py
├── strategies.py
├── simulator.py
├── evaluate.py
├── baselines.py
├── run_experiment.py
├── analyse.py
├── download_data.py
├── optimiser_template.py
├── prompts/
│   ├── system_a.txt
│   ├── system_b.txt
│   └── system_c.txt
└── controllers/
    ├── base.py
    ├── approach_a.py
    ├── approach_b.py
    ├── approach_c.py
    └── mock_controller.py
```

---

## Step 1: Download data (if not already done)

```bash
cd sdl_adaptive/
python download_data.py --data_dir data/
```

---

## Step 2: Run mock conditions (baseline benchmark)

This takes ~15–20 minutes and requires no Ollama.

```bash
python run_experiment.py \
    --data_dir data/ \
    --out_dir results/ \
    --n_seeds 20
```

Output: `results/metrics_summary.csv` with 820 rows
(9 conditions × 5 datasets × 20 seeds, where ada_original only runs on coatings).

---

## Step 3: Start Ollama server

In a separate terminal:
```bash
ollama serve
```

Leave this running throughout the LLM experiment.

---

## Step 4: Run real LLM conditions

This adds approach_a, approach_b, approach_c to the existing results.
Estimated runtime: **2–4 hours** (3 conditions × 5 datasets × 20 seeds × ~1–3 min/campaign).

```bash
python run_experiment.py \
    --data_dir data/ \
    --out_dir results/ \
    --n_seeds 20 \
    --llm_only \
    --model qwen2.5-coder:7b
```

The `--llm_only` flag runs only `approach_a`, `approach_b`, `approach_c` and
**merges** them into the existing `results/metrics_summary.csv` without overwriting
the mock-controller rows.

### Running a single dataset first (recommended sanity check)

```bash
python run_experiment.py \
    --data_dir data/ \
    --out_dir results_llm_test/ \
    --n_seeds 3 \
    --datasets coatings \
    --llm_only \
    --model qwen2.5-coder:7b
```

---

## Step 5: Regenerate figures with merged results

```bash
python analyse.py \
    --results_dir results/ \
    --out_dir figures/
```

This automatically picks up all conditions present in `metrics_summary.csv`,
including the new LLM conditions (labelled "LLM-A", "LLM-B", "LLM-C" in figures).

---

## Troubleshooting

### Ollama connection refused
```
Error: httpx.ConnectError: [Errno 111] Connection refused
```
→ Make sure `ollama serve` is running in a separate terminal.

### Model not found
```
Error: model 'qwen2.5-coder:7b' not found
```
→ Run `ollama pull qwen2.5-coder:7b` first.

### LLM returns invalid JSON (Approach A/B)
The controller retries up to 3 times with exponential backoff, then falls back to
the previous strategy. This is logged as a WARNING. A high fallback rate suggests
the model is not following the JSON format — try a larger model:
```bash
ollama pull qwen2.5-coder:14b
python run_experiment.py ... --model qwen2.5-coder:14b
```

### Approach C sandbox timeout
The subprocess sandbox has a 10-second timeout. If the LLM writes slow code
(e.g. fitting a GP with many restarts), it will time out and fall back to random.
This is expected behaviour and is recorded in `switch_logs.json` as a failure.

### Resuming an interrupted run
If the run is interrupted mid-way, re-run the same command. The merge logic in
`run_experiment.py` will overwrite only the conditions that were re-run, preserving
all previously completed seeds.

---

## Merging results from a different machine

If you ran mock conditions on one machine and LLM conditions on another:

```python
import pandas as pd

mock = pd.read_csv("results_mock/metrics_summary.csv")
llm  = pd.read_csv("results_llm/metrics_summary.csv")

# Keep only LLM conditions from the LLM run
llm_only = llm[llm['condition'].isin(['approach_a', 'approach_b', 'approach_c'])]

combined = pd.concat([mock, llm_only], ignore_index=True)

# Guard: drop any accidental duplicates on (dataset, condition, seed)
before = len(combined)
combined = combined.drop_duplicates(subset=["dataset", "condition", "seed"], keep="last")
if len(combined) < before:
    print(f"Warning: dropped {before - len(combined)} duplicate rows")

combined.to_csv("results_combined/metrics_summary.csv", index=False)
print(f"Combined: {len(combined)} rows, {combined['condition'].nunique()} conditions")
```

Then copy the per-condition subdirectories:
```bash
cp -r results_llm/coatings/approach_a results_combined/coatings/
cp -r results_llm/coatings/approach_b results_combined/coatings/
cp -r results_llm/coatings/approach_c results_combined/coatings/
# ... repeat for each dataset
```

---

## Expected results (mock controllers, for comparison)

From the 20-seed benchmark on real ADA data (derived from `metrics_summary.csv`):

**Coatings (7D, single-objective)**

| Condition | AUC mean ± SD | Final best (norm.) | Switch freq. |
|-----------|--------------|-------------------|-------------|
| Random | 0.886 ± 0.060 | 0.974 | 0.000 |
| LHS | 0.891 ± 0.046 | 0.974 | 0.000 |
| UCB β=0.2 | 0.914 ± 0.046 | 0.988 | 0.000 |
| UCB β=400 | 0.912 ± 0.036 | 0.973 | 0.000 |
| EI (fixed) | 0.910 ± 0.052 | 0.990 | 0.000 |
| ADA original | 0.886 ± 0.059 | 0.980 | 0.110 |
| Mock-A (β-tune) | 0.912 ± 0.044 | 0.990 | 0.000 |
| **Mock-B (switch)** | **0.921 ± 0.031** | **0.991** | 0.736 |
| Mock-C (code) | 0.780 ± 0.129 | 0.874 | 0.000 |

**Pareto (4D, multi-objective) — mean AUC across 4 campaigns**

| Condition | AUC mean ± SD |
|-----------|--------------|
| Random | 0.787 ± 0.124 |
| LHS | 0.794 ± 0.134 |
| UCB β=0.2 | 0.721 ± 0.150 |
| UCB β=400 | 0.782 ± 0.108 |
| EI (fixed) | 0.766 ± 0.125 |
| Mock-A (β-tune) | 0.752 ± 0.126 |
| Mock-B (switch) | 0.787 ± 0.130 |
| Mock-C (code) | 0.630 ± 0.160 |

**Key caveat on mock_approach_c**: This condition is highly seed-sensitive.
The mock implementation uses LHS with a very small X_obs, which occasionally
produces degenerate suggestions. In the original 20-seed run, 4/20 seeds hit
this failure mode (AUC < 0.7), pulling the mean down to 0.780 ± 0.129.
A fresh 20-seed run with different RNG initialisation drew 0/20 bad seeds,
giving 0.907 ± 0.064. This is not a bug — it reflects genuine instability in
the mock-C heuristic. With more seeds (e.g. 50+) the mean converges.
The real LLM-C (approach_c) may perform differently depending on what code
the LLM generates.

---

## Notes on scientific interpretation

- **NNOracle**: All results use nearest-neighbour lookup on real ADA measurements.
  The oracle never extrapolates — every returned value is a real experimental measurement.
  This means AUC is bounded in [0, 1] by construction.

- **20 seeds**: Each seed uses a different random initialisation order. The 91-point
  coatings dataset is small enough that seed variance is substantial (SD ~0.03–0.13).
  Interpret differences < 0.05 AUC with caution.

- **Pareto datasets**: The scalarised objective (equal-weight sum of z-scored conductance
  + conductivity) has negative values. Range-normalised AUC handles this correctly.
  Random search outperforms BO on 2/4 Pareto campaigns — this is a known low-data
  BO failure mode (GP poorly calibrated with < 65 points in 4D).
