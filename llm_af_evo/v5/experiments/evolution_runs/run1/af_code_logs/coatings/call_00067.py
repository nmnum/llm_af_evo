def score_pool(context):
    """Blend acquisition value with uncertainty-aware progress sensitivity and adaptive novelty penalty."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Normalize acq_value_norm to [0, 1] across pool (already provided)
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        aq = cand["acq_value_norm"]

        # Compute uncertainty as sum of normalized stds
        front_range = context["pareto_front_range"]
        sigma_sum = sum(gp[name]["std"] / (front_range[name] + 1e-9) for name in names)

        # Progress-aware exploitation: scale by campaign progress  
        prog = context["campaign"]["progress"]

        # Sensitivity to uncertainty increases as we get closer to end
        ucb_weight = np.clip(2.0 * (1 - prog), 0, 2)
        
        # Mix acquisition value with UCB-style exploration 
        exploit_score = aq + ucb_weight * sigma_sum

        scores.append(exploit_score)

    return scores