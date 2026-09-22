def score_pool(context):
    """Adaptive exploitation-exploration balance with uncertainty-aware hypervolume projection and progress-driven std scaling."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamic blend: start more exploitative, shift to explorative
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.4)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: normalized mean sum  
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Exploration: scaled uncertainty, modulated by progress and range
        sigma_scaled_sum = sum(
            gp[name]["std"] * (1.0 + 3.0 * np.exp(-4*progress)) / front_range[name]
            for name in names
        )
        
        # Combine with dynamic weight 
        score_exploit = w_exploit * mu_norm_sum  
        score_exploration = (1 - w_exploit) * sigma_scaled_sum
        
        scores.append(score_exploit + score_exploration)
    
    return scores