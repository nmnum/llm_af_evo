def score_pool(context):
    """Rank candidates by their distance in objective space to the nearest previously observed point."""
    names = context["objective_names"]
    y_obs = context["Y_obs"]
    
    scores = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to all observations
        dists = np.sqrt(np.sum((y_obs - gp_mean) ** 2, axis=1))
        min_dist = np.min(dists)
        
        scores.append(min_dist)

    return scores