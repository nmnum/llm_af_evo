# sdl-adaptive

A simulation benchmark for evaluating LLM-based optimisation strategy selection in self-driving laboratory (SDL) campaigns.

## Research question

> Given an experiment campaign that is already running, how should an autonomous system decide which optimisation strategy to use, and when to switch?

Rather than proposing new optimisation algorithms, this project systematises and automates the strategy-level decisions that are currently made heuristically during SDL campaigns. Three approaches are evaluated, each representing a different level of autonomy given to a language model controller.

---

## The three approaches

| | Approach A | Approach B | Approach C |
|---|---|---|---|
| **What the LLM controls** | A single parameter (β in UCB) | Which strategy to use | The entire optimiser as Python code |
| **LLM output** | A float | A strategy name + params (JSON) | A complete `suggest()` function |
| **Analogy** | Adjusting a dial | Choosing which tool to use | Writing a new tool from scratch |

All three approaches receive the same campaign context at each decision point: current step, budget, best result so far, improvement rate over the last 10 steps, and a GP uncertainty estimate.

---

## Repository structure

```
sdl_adaptive/
├── data/                        # Real experimental datasets (download separately)
│   ├── coatings_2022.csv        # 91 thin-film coating experiments, 7 dimensions
│   ├── pareto_20201218.csv      # 65 Pareto optimisation experiments, 4 dimensions
│   ├── pareto_20201223.csv      # 63 experiments, 4 dimensions
│   ├── pareto_20210104.csv      # 53 experiments, 4 dimensions
│   └── pareto_20210112.csv      # 72 experiments, 4 dimensions
│
├── oracle.py                    # Wraps a dataset as a queryable experiment oracle
├── strategies.py                # Six optimisation strategies (UCB, EI, PI, Thompson, random, LHS)
├── simulator.py                 # Step-by-step campaign simulator
├── baselines.py                 # Fixed strategies that never switch (comparison baselines)
│
├── controllers/
│   ├── base.py                  # Shared Ollama LLM interface
│   ├── approach_a.py            # LLM tunes UCB β
│   ├── approach_b.py            # LLM switches strategy
│   ├── approach_c.py            # LLM rewrites optimiser code (subprocess sandbox)
│   └── mock_controller.py       # Rule-based versions of all three (no LLM required)
│
├── prompts/
│   ├── system_a.txt             # System prompt for approach A
│   ├── system_b.txt             # System prompt for approach B
│   └── system_c.txt             # System prompt for approach C
│
├── optimiser_template.py        # Starting code given to approach C
├── run_experiment.py            # Main entry point
├── evaluate.py                  # Metrics: AUC, steps-to-threshold, switch count
├── analyse.py                   # Figures and statistical tests
├── download_data.py             # Downloads the ADA benchmark datasets
└── reproduce.ipynb              # Notebook reproducing all results
```

---

## Quickstart

### 1. Install dependencies

```bash
conda create -n sdl-adaptive python=3.11
conda activate sdl-adaptive
pip install numpy scipy scikit-learn pandas matplotlib seaborn
```

### 2. Download data

```bash
python download_data.py --out_dir data/
```

### 3. Run mock baselines (no LLM required)

```bash
python run_experiment.py \
  --data_dir data/ \
  --out_dir results/ \
  --n_seeds 20
```

### 4. Run real LLM approaches (requires Ollama)

Install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull qwen2.5-coder:7b
```

Then run:

```bash
python run_experiment.py \
  --data_dir data/ \
  --out_dir results/ \
  --n_seeds 20 \
  --llm_only \
  --model qwen2.5-coder:7b
```

### 5. Generate figures

```bash
python analyse.py --results_dir results/ --out_dir figures/
```

---

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `data/` | Directory containing dataset CSVs |
| `--out_dir` | `results/` | Output directory for metrics CSV |
| `--n_seeds` | `20` | Number of random seeds per condition |
| `--llm_only` | off | Run only real LLM approaches (skip mock/baselines) |
| `--discrete` | off | Use discrete scoring mode (no snap artefact) |
| `--model` | `qwen2.5-coder:7b` | Ollama model name |

---

## How a campaign simulation works

**Oracle.** Each dataset is wrapped by a neural-network oracle (`oracle.py`) that returns real measurements when queried. This avoids the extrapolation problem of a GP oracle (which produced values 193× the empirical maximum on the coatings dataset).

**Simulator.** One campaign proceeds as follows:
1. Initialise with 5 random experiments.
2. Every 5 steps, call the controller: "what strategy should I use now?"
3. Use that strategy to pick the next experiment.
4. Record the running best at each step.
5. Repeat until the budget (= dataset size) is exhausted.

**Discrete mode.** By default, continuous strategies suggest a point in ℝᵈ which is snapped to the nearest dataset row. This causes collisions (the same row queried multiple times) and reshuffles strategy rankings (Spearman ρ = 0.117 between snap and discrete rankings on coatings). With `--discrete`, strategies score only the unqueried rows directly - no snap, no collisions.

**Metrics.** For each campaign:
- `auc_best`: area under the normalised running-best curve (primary metric)
- `final_best_normalised`: fraction of global best found by end of campaign
- `steps_to_90pct`: steps to reach 90% of global best
- `switch_count`: number of strategy switches made by the controller
- `failure_count`: number of steps where the strategy raised an exception

---

## Baselines

Fixed strategies that never switch, used as comparison points:

| Condition | Description |
|-----------|-------------|
| `fixed_ucb_low` | UCB β=0.2 (pure exploitation) |
| `fixed_ucb_high` | UCB β=400 (pure exploration) |
| `fixed_ei` | Expected Improvement throughout |
| `fixed_lhs` | Latin Hypercube Sampling throughout |
| `fixed_random` | Uniform random throughout |
| `ada_original` | Replays the strategy used in the original ADA paper |

Mock controllers (rule-based, no LLM):

| Condition | Description |
|-----------|-------------|
| `mock_approach_a` | Heuristic β-tuning based on improvement rate |
| `mock_approach_b` | Fixed phase schedule: LHS → EI → UCB β=20 → UCB β=0.2 |
| `mock_approach_c` | Switches between inline implementations by phase |

---

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| NN oracle (not GP) | GP oracle extrapolated to 193× empirical max in 7D with 91 points; NN lookup always returns a real measurement |
| Random candidate grid (300 pts) | Replaces `scipy.optimize` for acquisition maximisation - simpler, faster, no shape ambiguity |
| Discrete scoring mode | Avoids snap artefact; Spearman ρ = 0.117 between snap and discrete rankings on coatings |
| Lazy GP uncertainty | GP fitting is slow (~2s/call); only computed when `controller._needs_gp_uncertainty = True` |
| Widened kernel bounds `(1e-3, 1e3)` | Prevents sklearn L-BFGS-B from hitting the default `1e-5` lower wall on spiky landscapes |

---

## Limitations

- **Small datasets.** N = 53–91 points. Seed variance is high (SD 0.03–0.14 AUC). Differences < 0.05 AUC should be interpreted cautiously.
- **Mock controllers are heuristics.** They are rule-based, not real LLM outputs. Their performance reflects specific implementation choices, not fundamental properties of each approach.
- **Single model tested.** Results are for `qwen2.5-coder:7b` via Ollama. Different models will produce different code quality and failure rates.
- **Continuous oracle landscape.** The NN oracle creates a discrete landscape - strategies that query the same region repeatedly will hit the same training point. This may understate the advantage of space-filling strategies.

---

## Data source

Datasets are from the ADA (Autonomous Discovery Accelerator) benchmark:

> Langner, S. et al. *Beyond Ternary OPV: High-Throughput Experimentation and Self-Driving Laboratories Optimize Multicomponent Systems.* Advanced Materials 2020.

Downloaded via `download_data.py` from the [ADA GitHub repository](https://github.com/PV-Lab/ADA).
