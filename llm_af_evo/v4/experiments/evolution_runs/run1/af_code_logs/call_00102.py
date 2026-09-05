def score_pool(context):
    """Reward candidates based on how far their predicted objectives are from any previously observed point in objective space."""
    scores = []
    y_obs = context["Y_obs"]
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
        
        # Compute distances to all past observations
        dists = np.linalg.norm(y_obs - gp_mean, axis=1)
        min_dist = np.min(dists)
        
        scores.append(min_dist)
    
    return scores