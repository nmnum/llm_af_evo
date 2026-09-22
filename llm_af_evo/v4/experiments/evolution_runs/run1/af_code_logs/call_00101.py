def score_pool(context):
    """Reward candidates with predicted outcomes farthest from any previously observed outcome in objective space."""
    scores = []
    y_obs = context["Y_obs"]
    
    for cand in context["pool"]:
        mu = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
        
        # Compute distances to all observations
        dists = np.linalg.norm(y_obs - mu, axis=1)
        
        # Use the minimum distance as novelty score (higher is better)
        scores.append(np.min(dists))
    
    return scores