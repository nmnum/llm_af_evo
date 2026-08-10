def score_pool(context):
    """Score candidates by the distance to their nearest observed neighbor in objective space, rewarding novelty."""
    scores = []
    y_obs = context["Y_obs"]
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
        
        # Compute distances to all observed points
        dists = np.linalg.norm(y_obs - gp_mean, axis=1)
        
        # Take the minimum distance (nearest neighbor)
        min_dist = np.min(dists)
        
        scores.append(min_dist)

    return scores