def score_pool(context):
    """Progress-aware exploitation with uncertainty-adjusted hypervolume signal and dynamic scaling."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Blend: more exploit as we near the end, but not too aggressively
    w_exploit = 0.3 + 0.7 * np.tanh(4 * (progress - 0.5))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: normalized mean of objectives 
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty term with dynamic scaling based on progress
        sigma_scaled = sum(
            gp[name]["std"] * (1.0 + 3.0 * np.exp(-2 * progress)) 
            for name in names
        )
        
        score = w_exploit * mu_sum_norm - (1.0 - w_exploit) * sigma_scaled
        
        scores.append(score)
    
    return scores