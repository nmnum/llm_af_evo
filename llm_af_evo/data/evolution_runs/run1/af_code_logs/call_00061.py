def score_pool(context):
    """Combine exploitation and novelty with progress-aware weighting."""
    X_obs = context["X_obs"]
    campaign = context["campaign"]
    front_range = context["pareto_front_range"]
    
    # Progress-aware exploitation weight: start low, increase towards end
    progress = campaign["progress"]
    exploit_weight = 0.3 + 0.7 * progress
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: weighted sum of means
        mu_sum = (gp["Tm"]["mean"] * front_range["Tm"] +
                  gp["kD"]["mean"] * front_range["kD"] +
                  gp["viscosity"]["mean"] * front_range["viscosity"])
        
        # Uncertainty: normalized sum of stds
        sigma_norm = (gp["Tm"]["std"] / front_range["Tm"] +
                      gp["kD"]["std"] / front_range["kD"] +
                      gp["viscosity"]["std"] / front_range["viscosity"])
        
        # Novelty: inverse distance to nearest observed point
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = 1.0 / (1e-8 + dists.min())
        
        # Combined score
        exploit_score = exploit_weight * mu_sum
        explore_score = (1 - exploit_weight) * sigma_norm
        novelty_score = 0.5 * novelty
        
        scores.append(exploit_score + explore_score + novelty_score)
    
    return scores