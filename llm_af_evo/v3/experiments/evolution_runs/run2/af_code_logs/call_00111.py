def score_pool(context):
    """Adapts exploration-exploitation balance using progress-aware sigmoid and incorporates normalized hypervolume contribution from reference point."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploitation early, exploration later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized means and uncertainties by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # Hypervolume contribution estimate based on distance to reference point
        hv_contribution = np.prod([max(0.0, ref_point[i] - gp[names[i]]["mean"]) 
                                   for i in range(len(names))])
        
        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * sigma_norm_sum + 0.5 * hv_contribution
        
        scores.append(score)
    
    return scores