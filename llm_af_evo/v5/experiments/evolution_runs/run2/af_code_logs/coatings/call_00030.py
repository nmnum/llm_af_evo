def score_pool(context):
    """Suppress acquisition scores for candidates that are close to already-selected front runners, promoting diverse exploration."""
    names = context["objective_names"]
    
    # Use a simple distance metric (normalized Euclidean) from each candidate's x to the nearest observed point  
    def novelty_score(cand_x):
        if len(context["X_obs"]) == 0:
            return float('inf')
        
        distances = np.linalg.norm(context["X_obs"] - cand_x, axis=1)
        min_distance = np.min(distances) 
        # Convert to a score: higher means more novel (farther from existing points).
        # Use inverse of distance with small epsilon for numerical stability
        return 1.0 / max(min_distance, 1e-8)

    scores = []
    
    acq_vals = [cand["acq_value_norm"] for cand in context["pool"]]
        
    if len(context["X_obs"]) == 0:
        # No observations yet; just use acquisition values
        return acq_vals
    
    novelty_scores = [novelty_score(cand["x"]) for cand in context["pool"]] 
    
    # Combine the two signals with a fixed weight (tuned to balance exploitation vs exploration)
    weights = np.array([0.6, 0.4])   # more emphasis on acquisition value
    combined_scores = []
    
    for i, _cand in enumerate(context["pool"]):
        score = weights[0] * acq_vals[i] + weights[1] * novelty_scores[i]
        
        if len(combined_scores) > 2: 
            recent_top_indices = np.argsort([combined_scores[j] for j in range(max(0, i-3),i)])[-2:]
            
            # If this candidate is very close to a top-ranked one (within threshold)
            distances_to_recent = [np.linalg.norm(context["pool"][j]["x"] - _cand["x"]) 
                                   for j in recent_top_indices if j != i]
                
            min_dist = np.min(distances_to_recent) if len(distances_to_recent)>0 else float('inf')
            
            # Suppress score of candidates too similar to top-ranked ones
            suppression_factor = 1. - (min_dist / max(0.2, context["pareto_front_range"][names[0]])) 
          
            suppressed_score = min(suppression_factor * score ,score)  
        else:
             suppressed_score = score
        
        combined_scores.append(max(0.,suppressed_score))
    
    return list(combined_scores)