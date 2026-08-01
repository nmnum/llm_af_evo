def score_pool(context):
    """Exploitation with adaptive uncertainty weighting and novelty penalty: balances mean prediction, uncertainty, and diversity from previously observed points."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Adaptive UCB weight: more uncertainty early, less later
        ucb_weight = 0.5 + 0.5 * progress
        
        # Uncertainty contribution
        sigma_sum = sum(gp[name]["std"] for name in names)
        
        # Novelty penalty: distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            novelty_penalty = np.min(distances)
        else:
            novelty_penalty = 0.0
        
        # Final score: exploitation + uncertainty bonus, minus novelty penalty
        scores.append(mu_sum + ucb_weight * sigma_sum - 0.1 * novelty_penalty)
    
    return scores