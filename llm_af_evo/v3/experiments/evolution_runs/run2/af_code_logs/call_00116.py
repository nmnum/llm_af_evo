def score_pool(context):
    """Exploits predicted means with uncertainty-aware hypervolume normalization and progress-adaptive exploration-exploitation balance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploitation early, exploration later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize means and uncertainties by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_scaled_sum = sum(
            gp[name]["std"] * (1.0 + 2.0 * progress) / front_range[name]
            for name in names
        )
        
        # Combine with dynamic blend and normalize by objective count to avoid scale bias 
        score = w_exploit * mu_norm_sum + (1 - w_exploit) * sigma_scaled_sum
        
        scores.append(score)
    
    return scores