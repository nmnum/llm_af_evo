def score_pool(context):
    """Blend acquisition value with inverse distance to nearest observation for novelty-driven exploration."""
    names = context["objective_names"]
    
    # Use normalized acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute squared Euclidean distances from each candidate x to all observations
    X_obs = context["X_obs"]
    candidates_x = np.array([cand["x"] for cand in context["pool"]])  # (n_pool, d)
    
    if len(X_obs) == 0:
        min_distances = np.full(len(candidates_x), float('inf'))
    else:
        # Broadcasting: each candidate x vs all observed X_obs
        diff_sq = candidates_x[:, None, :] - X_obs[None, :, :]
        distances_squared = np.sum(diff_sq**2, axis=2)  # (n_pool, n_obs)
        min_distances = np.min(distances_squared, axis=1)

    novelty_scores = []
    
    for i in range(len(candidates_x)):
        
        dist_to_nearest_observe = min_distances[i]
                
        if not np.isinf(dist_to_nearest_observe):
            # Invert distance (higher is better) and normalize
            norm_novelty_score = 1.0 / (dist_to_nearest_observe + 1e-8)
            
        else:
            # No observations yet, assign high novelty score  
            norm_novelty_score = 1.
        
        final_combined_score = acq_scores[i] * (
              0.75 * np.clip(norm_novelty_score / (np.max(min_distances) + 1e-8), 
                             a_min=0., a_max=None)
                )
                
        novelty_scores.append(final_combined_score)

    return novelty_scores