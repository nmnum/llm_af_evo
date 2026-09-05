def score_pool(context):
    """Balances exploitation and uncertainty with progress-aware weights, enhanced by novelty distance from observed points."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign_progress = context["campaign"]["progress"]

    # Adaptive weight: start more uncertainity-focused early, decrease as we approach the end
    w_uncert = 0.5 + 0.3 * np.sin(campaign_progress * np.pi / 2)
    w_exploit = 1 - w_uncert

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize means and stds
        mu_norm = sum(gp[name]["mean"] / front_range[name] for name in names) 
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Exploitation term (weighted mean), uncertainty term, novelty bonus  
        exploit_score = w_exploit * mu_norm
        uncert_score = w_uncert * sigma_norm
        
        # Novelty: inverse of minimum distance to any observed point 
        cand_x = cand["x"]
        distances = np.linalg.norm(X_obs - cand_x, axis=1)
        min_distance = float(distances.min())
        
        novelty_bonus = 0.5 / (min_distance + 1e-8) # avoid division by zero
        
        score = exploit_score + uncert_score + novelty_bonus
        scores.append(score)

    return scores