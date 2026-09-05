def score_pool(context):
    """Score candidates by inverse of nearest observed neighbor distance in objective space."""
    scores = []
    y_obs = context["Y_obs"]
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] 
                           for name in context["objective_names"]])
        
        # Compute distances to all observed points
        dists = np.linalg.norm(y_obs - gp_mean, axis=1)
        
        # Use the minimum distance as novelty score (higher is better)
        min_dist = np.min(dists) if len(dists) > 0 else float('inf')
        
        # Invert for scoring: larger distances -> higher scores
        score = 1.0 / (min_dist + 1e-8) 
        scores.append(score)

    return scores