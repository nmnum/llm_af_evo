def score_pool(context):
    """Adaptive exploitation-uncertainty tradeoff using progressive hypervolume signal scaling and normalized uncertainty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Use a smooth blend from pure exploration to exploitation based on campaign progress
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Compute hypervolume contribution estimate using reference point and normalized means
        hv_contrib = 1.0
        for name in names:
            mean_val = gp[name]["mean"] 
            ref_val = ref_point[list(names).index(name)]
            if mean_val > ref_val:  
                hv_contrib *= (mean_val - ref_val) / front_range[name]
        
        # Normalize uncertainty by the current range of objectives
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        score = w_exploit *hv_contrib + (1.0-w_exploit)*sigma_norm 
        scores.append(score)
    
    return scores