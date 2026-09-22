def score_pool(context):
    """Uncertainty-weighted hypervolume improvement estimate with progress-adaptive exploration-exploitation balance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Progress-aware exploitation weight
    w_exploit = 0.3 + 0.7 * np.tanh(2 * (progress - 0.5))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted mean and std normalized by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_scaled_sum = sum(
            gp[name]["std"] * (1.0 + 2.5 * progress) / front_range[name]
            for name in names
        )
        
        # Combine exploitation and uncertainty terms with dynamic weight 
        score_exploit = w_exploit * mu_norm_sum  
        score_uncertainty = (1 - w_exploit) * sigma_scaled_sum
        
        scores.append(score_exploit + score_uncertainty)
    
    return scores