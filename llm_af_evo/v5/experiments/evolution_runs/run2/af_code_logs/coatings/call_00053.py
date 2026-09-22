def score_pool(context):
    """Blend acquisition value with inverse distance to nearest observed point for progress-aware exploration."""
    names = context["objective_names"]
    
    # Use normalized acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute distances from each candidate to the closest previously observed point
    X_obs = context["X_obs"]
    if len(X_obs) == 0:
        min_distances = np.full(len(context["pool"]), float('inf'))
    else:
        candidates_x = np.array([cand['x'] for cand in context["pool"]])
        # Vectorized Euclidean distances
        diff = X_obs[:, None, :] - candidates_x[None, :, :]
        dists_squared = np.sum(diff**2, axis=2)
        min_distances_sq = np.min(dists_squared, axis=0)  
        min_distances = np.sqrt(min_distances_sq)

    # Invert distance (closer = higher score), handle case where distance is zero
    novelty_scores = 1. / (min_distances + 1e-8) 

    # Normalize to [0, 1] range for blending 
    max_novelty = np.max(novelty_scores)
    if max_novelty > 0:
        normalized_novelty = novelty_scores / max_novelty
    else:  
        normalized_novelty = novelty_scores

    final_scores = []
    
    # Blend acquisition score with novelty, weighting more towards exploitation early,
    # and exploration later as progress increases (a simple linear schedule)
    campaign_progress = context["campaign"]["progress"]
        
    for i in range(len(acq_scores)):
            
        w_acq = 0.7 + 0.3 * campaign_progress   # Start with high acquisition weight, decrease over time
        combined_score = w_acq * acq_scores[i] + (1 - w_acq) * normalized_novelty[i]
                
        final_scores.append(combined_score)
        
    return final_scores