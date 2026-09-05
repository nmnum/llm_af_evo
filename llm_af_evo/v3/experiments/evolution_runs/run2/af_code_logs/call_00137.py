def score_pool(context):
    """Progress-aware hypervolume-improvement estimate with dynamic exploitation-uncertainty balance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploration early, exploit later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted objective values normalized by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty scaled inversely with progress and objectivewise
        sigma_scaled = sum(
            gp[name]["std"] * (1.0 + 2.0 * np.exp(-5 * progress)) / front_range[name]
            for name in names
        )

        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * sigma_scaled
        
        scores.append(score)
    
    return scores