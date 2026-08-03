#!/bin/bash
# Step 3: Test whether n_init=15 fixes the routing signal quality problem.
#
# Hypothesis: with 15 init points instead of 5, the GP lengthscale estimate
# is reliable enough for the router to correctly identify structured landscapes
# and route to EGBO consistently.
#
# Run only on hartmann6 (largest EGBO advantage: Δ=0.167, p=0.005) and
# pareto_20210112 (confirmed structured landscape from power run).
#
# Expected runtime: ~45 minutes (egbo + rule_router fast; approach_d ~30 min)

python shared_seed_experiment.py \
    --data_dir data/ \
    --out_dir results_ninit15/ \
    --n_repeats 20 \
    --budget_frac 0.5 \
    --n_init 15 \
    --datasets hartmann6 pareto_20210112 \
    --conditions egbo rule_router approach_d fixed_random \
    --model qwen2.5:14b-instruct

echo ""
echo "Run conditioned analysis:"
echo "python mid_campaign_analysis.py --results_dir results_ninit15/ --threshold 0.80"
echo ""
echo "Run decision analysis:"
echo "python analyse_decisions.py --results_dir results_ninit15/ --condition approach_d --verbose"
echo ""
echo "Compare with n_init=5 results:"
echo "  hartmann6  n_init=5:  EGBO=0.791, approach_d=0.599 (conditioned)"
echo "  hartmann6  n_init=15: EGBO=?, approach_d=?"
echo ""
echo "If approach_d closes the gap with EGBO at n_init=15:"
echo "  -> Signal quality was the limiting factor"
echo "  -> Fix: always use n_init>=15 before first routing call"
echo ""
echo "If approach_d still underperforms EGBO at n_init=15:"
echo "  -> Switching cost or LLM capability is the limiting factor"
echo "  -> Fix: prior-knowledge routing (not observation-based)"
