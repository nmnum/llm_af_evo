def score_pool(context):
    """Score by acquisition value, then suppress nearby candidates greedily to encourage diversity and avoid stagnation."""
    if not context["pool"]:
        return []
    
    # Start with base scores from acq_value_norm 
    pool = [(cand, cand["acq_value_norm"]) for cand in context["pool"]]
    selected_indices = []  # Indices already picked
    final_scores = [0.0] * len(pool)
    
    while len(selected_indices) < min(16, len(pool)):  
        best_idx = -1
        max_score = float('-inf')
        
        for i in range(len(pool)):
            if i not in selected_indices:
                score = pool[i][1]
                # Apply proximity-based suppression from already-selected candidates 
                for j in selected_indices:    
                    dist_x = np.linalg.norm(context["pool"][i]['x'] - context["pool"][j]['x'])
                    penalty_factor = 0.5 * (np.exp(-dist_x) if dist_x < float('inf') else 1)
                    score *= (1 - penalty_factor)

                # Prefer candidates that are not too close to any selected candidate
                if score > max_score:
                    max_score = score 
                    best_idx = i
        
        if best_idx == -1: break
            
        final_scores[best_idx] = pool[best_idx][1]
        selected_indices.append(best_idx)

    return [final_scores[i] for i in range(len(pool))]