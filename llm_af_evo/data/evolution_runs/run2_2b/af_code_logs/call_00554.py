def score_pool(context):
    """Combine exploitation and novelty with progress-aware weighting."""
    X_obs = context["X_obs"]
    progress = context["campaign"]["progress"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: sum of means
        exploit_score = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        
        # Novelty: inverse distance to nearest observed point
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty_score = 1.0 / (dists.min() + 1e-8)
        
        # Progress-aware blending: early exploration, late exploitation
        w_exploit = 0.3 + 0.7 * progress
        score = w_exploit * exploit_score + (1 - w_exploit) * novelty_score
        
        scores.append(score)
    
    return scores