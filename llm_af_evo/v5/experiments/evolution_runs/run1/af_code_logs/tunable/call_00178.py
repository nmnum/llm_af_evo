def score_pool(context):
    """Score candidates by inverse distance to nearest observed point in objective space."""
    names = context["objective_names"]
    y_obs = context["Y_obs"]
    
    scores = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to all observed points
        dists = np.sqrt(np.sum((y_obs - gp_mean) ** 2, axis=1))
        min_dist = np.min(dists)
        
        # Invert distance as score (larger means more novel)
        if min_dist == 0:
            scores.append(float('inf'))
        else:
            scores.append(1. / min_dist)

    return scores