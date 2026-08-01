def score_pool(context):
    """Combine exploitation and novelty with progress-aware weighting."""
    X_obs = context["X_obs"]
    campaign = context["campaign"]
    front_range = context["pareto_front_range"]
    
    # Progress-aware weight for uncertainty vs exploitation
    p = campaign["progress"]
    w_exploit = 0.3 + 0.7 * p  # Start with more exploitation, shift towards uncertainty
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: sum of means
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        
        # Uncertainty: normalized std sum
        sigma_norm = (gp["Tm"]["std"] / front_range["Tm"] +
                      gp["kD"]["std"] / front_range["kD"] +
                      gp["viscosity"]["std"] / front_range["viscosity"])
        
        # Novelty: inverse distance to nearest observed point
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = 1.0 / (1e-8 + dists.min())
        
        # Combined score
        score = w_exploit * mu_sum + (1 - w_exploit) * sigma_norm + 0.1 * novelty
        scores.append(score)
    
    return scores