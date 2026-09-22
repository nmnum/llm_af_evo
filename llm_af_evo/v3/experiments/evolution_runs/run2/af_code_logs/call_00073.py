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
        
        # Exploitation signal based on distance to reference point
        mu_distances = [ref_point[i] - gp[name]["mean"] 
                        for i, name in enumerate(names)]
        hypervolume_diff = sum(d if d > 0 else 0 for d in mu_distances)
                
        # Uncertainty term scaled by progress and inverse of mean performance  
        sigma_weighted = sum(
            gp[name]["std"] * (1.0 + np.exp(-5 * progress)) / max(1e-6, gp[name]["mean"])
            for name in names
        )

        score = w_exploit * hypervolume_diff - (1.0 - w_exploit) * sigma_weighted  
        scores.append(score)
    
    return scores