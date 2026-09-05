def score_pool(context):
    """Combines progressive exploitation with uncertainty-sensitive hypervolume gain estimation using ref_point-based normalization."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Early: favor candidates that could improve HV significantly; later: balance exploitation and uncertainty
    w_exploit = 0.3 + 0.7 * np.tanh(2 * (progress - 0.5))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Estimate hypervolume gain potential based on distance from ref_point
        hv_potential = sum(
            max(ref_point[i] - gp[name]["mean"], 0) * (1.0 + progress)
            if i == 0 else 
            max(ref_point[i] - gp[name]["mean"], 0) / (1.0 + progress)
            for i, name in enumerate(names)
        )
        
        # Normalize uncertainty relative to the ref point and front range
        sigma_norm = sum(
            gp[name]["std"] * (
                np.log(1e-6 + abs(ref_point[i] - gp[name]["mean"])) / 
                (front_range[name])
            ) if i == 0 else  
            gp[name]["std"]
            for i, name in enumerate(names)
        )

        score = w_exploit * hv_potential + (1.0 - w_exploit) * sigma_norm
        scores.append(score)

    return scores