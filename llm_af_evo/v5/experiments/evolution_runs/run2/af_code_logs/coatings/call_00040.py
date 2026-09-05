def score_pool(context):
    """Suppress candidates that are too similar to already-selected points, encouraging exploration while preserving high-acquisition-value picks."""
    names = context["objective_names"]
    
    # Start with acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Track which candidates are suppressed due to proximity
    suppress_mask = [False] * len(context["pool"])
    
    if len(context["X_obs"]) > 0:
        X_obs = context["X_obs"]
        
        # For each candidate, compute min distance to any previously observed point  
        cand_dists_to_observed = []
        for i, cand in enumerate(context["pool"]):
            x_cand = cand["x"]
            
            if len(X_obs) > 0:
                dists = np.sum((X_obs - x_cand)**2, axis=1)
                min_dist_sq = np.min(dists)
                
                # Normalize by input space dimensionality (assumed to be [0, 1]^d)
                normalized_min_dist = min_dist_sq / len(x_cand) 
            else:
                normalized_min_dist = float('inf')
            
            cand_dists_to_observed.append(normalized_min_dist)

        # Suppress candidates that are too close (< threshold), prioritizing higher acquisition
        dist_threshold = 0.1
        
        for i in range(len(context["pool"])):
            if not suppress_mask[i] and cand_dists_to_observed[i] < dist_threshold:
                # Check neighbors: among all suppressed or un-suppressed candidates,
                # only keep the one with highest acq value
                candidate_indices = list(range(i+1, len(context["pool"])))+list(range(0,i))
                
                for j in range(len(candidate_indices)):
                    idx = candidate_indices[j]
                    
                    if not suppress_mask[idx] and cand_dists_to_observed[idx]<dist_threshold:
                        # Keep the one with higher acq score
                        keep_idx = i if (acq_scores[i]>acq_scores[idx]) else idx
                        
                        # Suppress the other 
                        to_suppress = [idx, i][keep_idx == i]
                        
                        suppress_mask[to_suppress] = True

    final_scores = []
    
    for i in range(len(context["pool"])):
        score = acq_scores[i]

        if not suppress_mask[i]:
            final_scores.append(score)
        else:
            # Penalize suppressed candidates
            penalty_factor=0.1 
            final_scores.append(penalty_factor * score)

    return final_scores