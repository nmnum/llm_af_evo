def score_pool(context):
    """Adaptive hypervolume signal with dynamic exploitation-uncertainty trade-off and progress-sensitive uncertainty scaling."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploration early, exploit later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized exploitation signal: mean of objectives scaled by front range
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty term with adaptive scaling based on progress and objective ranges  
        sigma_scaled = sum(
            gp[name]["std"] * (1.0 + 2.0 * np.exp(-5 * progress)) / front_range[name]
            for name in names
        )

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_scaled
        
        # Add a hypervolume-based signal derived from reference point and candidate's predicted objectives 
        hv_contribution = 1.0
        for i, name in enumerate(names):
            if gp[name]["mean"] > ref_point[i]:
                hv_contribution *= (gp[name]["mean"] - ref_point[i]) / front_range[name]
        
        score += 2.0 * np.exp(-5 * progress) *hv_contribution
        
        scores.append(score)
    
    return scores