def modifier(context):
    """Penalize candidates that are close to higher-ranked already-selected candidates, discouraging batch duplication."""
    if len(context["X_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    # Use a greedy approach: for each candidate, compute the minimum distance 
    # to any previously observed point in X_obs that has an acq_value_norm higher than this one
    values = []
    x_candidates = np.array([cand["x"] for cand in context["pool"]])
    obs_x = context["X_obs"]
    
    for i, cand in enumerate(context["pool"]):
        # Get the acquisition value of current candidate 
        acq_value_norm = cand["acq_value_norm"]

        min_dist_to_higher_acq = float('inf')
        
        # Iterate through all previously observed points
        for j, obs_point in enumerate(obs_x):  
            if context["Y_obs"][j][0] > 1e-6:  # Skip near-zero acquisition values (or use a small threshold)
                dist_sq = np.sum((x_candidates[i] - obs_point) ** 2)
                
                # Only consider points that were selected with higher acq_value_norm
                if context["Y_obs"][j][0] > acq_value_norm:
                    min_dist_to_higher_acq = min(min_dist_to_higher_acq, dist_sq)

        penalty = -min(1.0, 2 * np.sqrt(np.minimum(min_dist_to_higher_acq, float('inf')))) if not (np.isinf(min_dist_to_higher_acq) or min_dist_to_higher_acq == 0.) else 0.
        
        values.append(penalty)
    
    return values