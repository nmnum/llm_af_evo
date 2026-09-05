def score_pool(context):
    """Exploitation-uncertainty balance with progress-aware hypervolume-normalized uncertainty and dynamic blending."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Blend exploitation and uncertainty based on progress
    w_exploit = 0.3 + 0.7 * np.tanh(4 * (progress - 0.5))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Dynamic uncertainty with exponential decay and progress-aware scaling
        sigma_scaled = 0.5 * sum(
            (1 + np.exp(-3*progress)) * gp[name]["std"] / front_range[name]
            for name in names
        )
        
        score = w_exploit * mu_sum_norm + (1 - w_exploit) * sigma_scaled
        
        scores.append(score)
    
    return scores