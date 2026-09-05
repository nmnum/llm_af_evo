def score_pool(context):
    """Scores candidates based on predicted objective means and uncertainty scaled by front range, with progress-adaptive exploitation blending and square-root transformed uncertainty to encourage diverse exploration."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]

    # Sigmoidal blend: more exploitation early, exploration later
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize means and uncertainties by front range 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_sqrt_sum = np.sqrt(sum(gp[name]["std"] / front_range[name] for name in names))

        score = w_exploit * mu_norm_sum + (1.0 - w_exploit) * sigma_sqrt_sum
        
        scores.append(score)
    
    return scores