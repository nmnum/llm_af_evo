def score_pool(context):
    """Incorporate uncertainty-driven exploration bias into acquisition scores, favouring candidates with high expected improvement and low observed frequency."""
    names = context["objective_names"]
    
    # Compute base acq values for all candidates  
    acqs = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Estimate candidate novelty via inverse of squared distance to nearest observation
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        x_cand = cand["x"]
        
        if len(X_obs) == 0:
            novel_score = 1.0
        else:
            # Compute distances from candidate to all observations  
            dists_sq = np.sum((X_obs - x_cand)**2, axis=1)
            
            # Avoid division by zero; use min distance + small epsilon 
            nearest_dist_sq = np.min(dists_sq) if len(dists_sq) > 0 else 1.0
            novel_score = 1.0 / (nearest_dist_sq + 1e-8)

        acq = cand["acq_value_norm"]
        
        # Blend acquisition value with novelty score, weighted by uncertainty 
        ucb_bonus = sum(cand["gp_posterior"][name]["std"] for name in names)
        scores.append(acq * novel_score + 0.5 * ucb_bonus)

    return scores