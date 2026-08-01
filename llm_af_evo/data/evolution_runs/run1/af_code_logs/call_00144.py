def score_pool(context):
    """Combine exploitation and novelty with progress-aware weighting."""
    X_obs = context["X_obs"]
    progress = context["campaign"]["progress"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: sum of means
        exploit = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        
        # Novelty: distance to nearest observed point
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novel = float(dists.min())
        
        # Progress-aware weights: explore early, exploit later
        w_exploit = 0.3 + 0.7 * progress
        w_novel = 1.0 - w_exploit
        
        scores.append(w_exploit * exploit + w_novel * novel)
    
    return scores