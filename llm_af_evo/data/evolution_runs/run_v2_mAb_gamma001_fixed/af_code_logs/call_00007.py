def score_pool(context):
    """Score candidates by base acquisition value, then greedily reduce scores of nearby rivals to avoid batch redundancy."""
    names = context["objective_names"]
    pool = context["pool"]
    
    # Compute initial base scores using mean objectives (already maximised)
    base_scores = [sum(cand["gp_posterior"][name]["mean"] for name in names) 
                   for cand in pool]
    
    picked_indices = []
    final_scores = [-float('inf')] * len(pool)

    while len(picked_indices) < min(10, len(pool)):  # batch size limit
        best_idx = -1
        best_score = -float('inf')
        
        # Find highest base score among unpicked candidates  
        for i in range(len(pool)):
            if i not in picked_indices and base_scores[i] > best_score:
                best_score = base_scores[i]
                best_idx = i

        if best_idx == -1: break  # No more candidates
        
        picked_indices.append(best_idx)
        
        # Score this candidate as-is
        final_scores[best_idx] = base_scores[best_idx]

        # Reduce scores of nearby unpicked candidates  
        x_best = pool[best_idx]["x"]
        for i in range(len(pool)):
            if i not in picked_indices:
                dist_x = np.linalg.norm(x_best - pool[i]["x"])
                
                multiplier = 1.0 - np.exp(-dist_x)
                final_scores[i] = base_scores[i] * max(0, multiplier)

    # Handle any remaining unpicked candidates
    for idx in range(len(pool)):
        if idx not in picked_indices:
            final_scores[idx] = base_scores[idx]

    return final_scores