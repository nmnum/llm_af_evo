def score_pool(context):
    """Progress-sensitive exploitation with uncertainty-aware hypervolume gain estimation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Early exploration via UCB-like strategy, shift to pure exploitation later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.3)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: normalized mean 
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty term with dynamic scaling based on progress and objective ranges  
        sigma_scaled = sum(
            gp[name]["std"] * (1.0 + 3.0 * np.exp(-2 * progress)) / front_range[name]
            for name in names
        )

        score = w_exploit * mu_sum_norm + (1 - w_exploit) * sigma_scaled
        
        scores.append(score)
    
    return scores