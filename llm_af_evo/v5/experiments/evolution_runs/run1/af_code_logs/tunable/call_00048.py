def score_pool(context):
    """Rank candidates by their distance in objective space to the nearest previously observed point."""
    names = context["objective_names"]
    y_obs = context["Y_obs"]
    scores = []
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to all observations
        dists = np.linalg.norm(y_obs - gp_mean, axis=1)
        
        # Use the minimum distance as novelty score (higher is better)  
        scores.append(-np.min(dists))
    
    return scores