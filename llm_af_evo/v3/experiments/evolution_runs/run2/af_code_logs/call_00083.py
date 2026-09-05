def score_pool(context):
    """Exploits predicted objective means and uncertainty scaled by front range with dynamic blend controlled by progress-aware sigmoidal function."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blending: more exploitation early, exploration later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized means and uncertainties by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * np.sqrt(sigma_norm_sum)
        
        scores.append(score)
    
    return scores