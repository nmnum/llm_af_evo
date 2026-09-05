def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a novelty bonus for diverse exploration."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    
    # Compute squared distances to nearest observed point for each candidate
    X_obs = context["X_obs"] 
    novelties = []
    for cand in context["pool"]:
        x_cand = cand["x"]
        if len(X_obs) == 0:
            novelty = 1.0
        else:
            # Use squared Euclidean distance to nearest neighbor among observed points  
            dists_sq = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist_sq = np.min(dists_sq)
            # Convert back to normalised scale [0, 1] where larger means more novel
            novelty = 1.0 / (1.0 + min_dist_sq) 
        novelties.append(novelty)

    for i, cand in enumerate(context["pool"]):
        acq_norm = cand["acq_value_norm"]
        # Add small weighted novelty bonus to encourage exploration of unvisited regions
        score = 0.95 * acq_norm + 0.05 * novelties[i]
        scores.append(score)
        
    return scores