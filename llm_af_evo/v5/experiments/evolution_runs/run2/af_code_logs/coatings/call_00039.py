def score_pool(context):
    """Score candidates by blending acquisition value with a novelty bonus that penalizes similarity to already-observed points."""
    names = context["objective_names"]
    
    # Use normalized acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute squared Euclidean distances from each candidate's features 
    X_obs = context["X_obs"]
    candidates_x = np.array([cand["x"] for cand in context["pool"]]) 
    
    if len(X_obs) == 0:
        novelty_scores = np.zeros(len(candidates_x))
    else:    
        # For each candidate, find the minimum squared Euclidean distance to any observed point
        distances_sq = []
        
        for x_candidate in candidates_x:
            dists_to_observed = np.sum((X_obs - x_candidate)**2, axis=1)
            min_dist_sq = np.min(dists_to_observed) 
            distances_sq.append(min_dist_sq)

        # Normalize novelty scores to [0, 1] range
        max_distance = np.max(distances_sq)
        
        if max_distance > 0:
            novelty_scores = (max_distance - np.array(distances_sq)) / max_distance  
        else:   
            novelty_scores = np.zeros(len(candidates_x))

    # Combine acquisition score with normalized novelty bonus 
    final_scores = acq_scores + 0.3 * novelty_scores
    
    return list(final_scores)