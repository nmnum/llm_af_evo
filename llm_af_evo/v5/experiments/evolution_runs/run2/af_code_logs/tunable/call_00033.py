def score_pool(context):
    """Suppress candidates that are too similar to already-obtained points, favoring diversity in feature space while preserving high acquisition value."""
    scores = []
    
    # Use the normalized acquisition values as base 
    acq_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    X_obs = context["X_obs"]
    threshold = 0.1
    
    if len(X_obs) == 0:
        return acq_scores
        
    # Compute feature distances from each candidate to all observed points
    obs_dists = []
    for i, cand in enumerate(context["pool"]):
        x_cand = cand["x"]
        
        dists_to_obs = np.sum((X_obs - x_cand)**2, axis=1)
        min_dist = np.min(dists_to_obs) 
        
        # Normalize by number of features (assuming [0, 1] normalized input space)
        obs_dists.append(min_dist / len(x_cand))
        
    for i in range(len(context["pool"])):
        score = acq_scores[i]
            
        if obs_dists[i] < threshold:
            # Penalize candidates that are too close to existing points
            penalty_factor = 1.0 - (obs_dists[i]/threshold)
            score *= max(0.5, 1.0 - penalty_factor * 2) 
            
        scores.append(score)

    return scores