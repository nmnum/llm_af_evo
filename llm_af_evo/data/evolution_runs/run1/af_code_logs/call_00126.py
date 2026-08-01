def score_pool(context):
    """Combine exploitation and novelty with progress-aware weighting."""
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: weighted sum of means
        mu_sum = (gp["Tm"]["mean"] / front_range["Tm"] +
                  gp["kD"]["mean"] / front_range["kD"] +
                  gp["viscosity"]["mean"] / front_range["viscosity"])
        
        # Uncertainty: normalized standard deviation
        sigma_norm = (gp["Tm"]["std"] / front_range["Tm"] +
                      gp["kD"]["std"] / front_range["kD"] +
                      gp["viscosity"]["std"] / front_range["viscosity"])
        
        # Novelty: inverse distance to nearest observed point
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = 1.0 / (dists.min() + 1e-8)
        
        # Progress-aware weights
        exploit_weight = 0.5 + 0.5 * progress
        explore_weight = 1.0 - exploit_weight
        
        score = exploit_weight * mu_sum + explore_weight * sigma_norm + novelty
        scores.append(score)
    
    return scores