#!/bin/bash
# Sweep first_control_step on structured (hartmann6) and flat (pareto_20201218)
# to find the crossover where delayed routing beats immediate routing.
#
# Conditions per run: egbo, rule_router, approach_d, fixed_lhs, fixed_random
# Datasets: hartmann6 (EGBO wins by Δ=0.167), pareto_20201218 (random wins)
# Seeds: 20 per condition
# first_control_step ∈ {5, 10, 15, 20} (n_init=5, so 5=immediate)
#
# Runtime: ~10 min per first_control_step (no LLM for egbo/rule/lhs/random;
#           approach_d adds ~20 min per sweep point)
# Total: ~2 hours for all 4 sweep points × 2 datasets

MODEL="qwen2.5:14b-instruct"
DATASETS="hartmann6 pareto_20201218"
CONDITIONS="egbo rule_router approach_d fixed_lhs fixed_random"

for FCS in 5 10 15 20; do
    echo "========================================"
    echo "first_control_step = $FCS"
    echo "========================================"
    python shared_seed_experiment.py \
        --data_dir data/ \
        --out_dir results_fcs_sweep/fcs${FCS}/ \
        --n_repeats 20 \
        --budget_frac 0.5 \
        --n_init 5 \
        --first_control_step ${FCS} \
        --datasets ${DATASETS} \
        --conditions ${CONDITIONS} \
        --model ${MODEL}
done

echo ""
echo "Sweep complete. Analysing results..."
python analyse_fcs_sweep.py --sweep_dir results_fcs_sweep/
