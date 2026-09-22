def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a novelty bonus for diversity."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    scores = []
    
    # Compute distances from each candidate to the nearest observed point
    novelties = []
    for cand in context["pool"]:
        x_cand = cand["x"] 
        if len(X_obs) == 0:
            novelty = 1.0  
        else:
            dists = np.linalg.norm(X_obs - x_cand, axis=1)
            min_dist = np.min(dists)
            # Normalise by the range of features
            feature_range = np.max(X_obs, axis=0) - np.min(X_obs, axis=0)
            if np.any(feature_range == 0):
                novelty = 1.0  
            else:
                norm_min_dist = min_dist / np.mean(feature_range)
                # Invert so larger distances (more novel points) get higher scores
                novelty = max(0., 1. - norm_min_dist) 
        novelties.append(novelty)

    for i, cand in enumerate(context["pool"]):
        acq_norm = cand["acq_value_norm"]
        # Use a small weight on the novelty bonus (e.g. 5%)
        score = acq_norm + 0.05 * novelties[i]
        scores.append(score)

    return scores