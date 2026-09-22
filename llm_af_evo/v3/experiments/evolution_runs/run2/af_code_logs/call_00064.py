def score_pool(context):
    """Progress-adaptive exploitation-uncertainty blend with multiplicative uncertainty scaling and reference-point-normalized mean signal."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"] 
    progress = context["campaign"]["progress"]

    # Sigmoidal exploit/Explore trade-off
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize mean by reference point to encourage pushing beyond current front
        mu_ref_norm = sum(
            max(0, gp[name]["mean"] - ref_point[i]) / (ref_point[i] + 1e-8)
            for i, name in enumerate(names)
        )
                
        # Adaptive uncertainty: scale more strongly early, less later  
        u_scale_factor = np.exp(-3 * progress) 
        sigma_scaled = sum(
            gp[name]["std"] * u_scale_factor / front_range[name]
            for name in names
        )

        score = w_exploit * mu_ref_norm + (1.0 - w_exploit) * sigma_scaled  
        scores.append(score)
    
    return scores