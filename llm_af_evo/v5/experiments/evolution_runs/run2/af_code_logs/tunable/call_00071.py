def score_pool(context):
    """Blend acquisition value with an adaptive uncertainty-reward that scales based on front sparsity and campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context['ref_point_by_name'][name] for name in names])
    
    # Compute average front density (inverse of range spread)
    avg_density = 1.0 / np.mean(list(front_range.values()))
    
    # Adaptive uncertainty weight based on campaign progress and front sparsity
    progress = context["campaign"]["progress"]
    ucb_weight = max(0.2, 1.0 - progress) * (avg_density + 0.5)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Use acquisition value as main signal
        score = cand['acq_value_norm']
        
        # Add uncertainty bonus scaled by front density and campaign progress 
        sigma_sum = sum(gp[name]["std"] for name in names)
        ucb_bonus = ucb_weight * sigma_sum
        
        scores.append(score + 0.5 * ucb_bonus)

    return scores