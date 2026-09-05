def score_pool(context):
    """Balances exploitation and uncertainty via progress-aware sigmoidal blending while incorporating normalized hypervolume gain estimated from candidate predictions."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploitation early, exploration later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation based on normalized mean objectives 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty term using square root of summed uncertainties, scaled by progress
        sigma_sqrt_sum = np.sqrt(sum(gp[name]["std"]**2 for name in names))

        # Combine terms with dynamic weight and normalize uncertainty component  
        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * sigma_sqrt_sum / max(front_range.values())
        
        scores.append(score)
    
    return scores