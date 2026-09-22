def score_pool(context):
    """Progress-aware exploitation with uncertainty-weighted hypervolume difference from reference point."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploration early, exploit later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation signal based on distance to reference point, normalized
        mu_distances = [ref_point[i] - gp[name]["mean"] for i, name in enumerate(names)]
        mu_sum_norm = sum(dist / (1.0 + ref_point[i]) for i, dist in enumerate(mu_distances))
        
        # Uncertainty term scaled by progress and objective ranges  
        sigma_scaled = sum(
            gp[name]["std"] * np.exp(-2 * progress) 
            for name in names
        )

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_scaled
        scores.append(score)
    
    return scores