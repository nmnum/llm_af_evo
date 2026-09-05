def score_pool(context):
    """Score candidates by base acquisition value, then greedily reduce scores of nearby candidates to encourage diversity."""
    if not context["pool"]:
        return []
    
    # Use acq_value_norm as the base score for each candidate.
    pool = [(cand, cand["acq_value_norm"]) for cand in context["pool"]]
    selected_indices = []  # Indices already picked
    final_scores = [0.0] * len(pool)
    
    while len(selected_indices) < min(16, len(pool)):  # Batch size limit: pick up to 16 candidates.
        best_idx = -1
        max_score = float('-inf')
        
        for i in range(len(pool)):
            if i not in selected_indices:
                score = pool[i][1] * get_multiplier(i, selected_indices, context)
                if score > max_score:
                    max_score = score
                    best_idx = i
        
        # If no valid candidate left to pick.
        if best_idx == -1: break
            
        final_scores[best_idx] = pool[best_idx][1]
        selected_indices.append(best_idx)

    return [final_scores[i] for i in range(len(pool))]

def get_multiplier(i, picked_indices, context):
    """Compute a multiplier based on distance to already-picked candidates."""
    if not picked_indices:
        return 1.0
    
    cand = context["pool"][i]
    
    # Compute minimum Euclidean distances from this candidate's x
    min_dist_x = float('inf')
    for j in picked_indices:
        other_cand = context["pool"][j] 
        dist = np.linalg.norm(cand['x'] - other_cand['x'])
        if dist < min_dist_x:  
            min_dist_x = dist
    
    # Use the same formula as described.
    return 1.0 - np.exp(-min_dist_x)