def score_pool(context):
    """Score candidates by base acquisition value with greedy diversity: first pick best candidate, then reduce scores of nearby contenders in feature space."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Start with acq_value_norm as the initial scoring signal (already properly scaled)
    scores = [cand['acq_value_norm'] for cand in context["pool"]]
    selected_indices = []
    final_scores = [0.0] * pool_size
    
    # Greedy selection loop
    while len(selected_indices) < pool_size:
        # Pick the best remaining candidate by current score (not yet finalized)
        idx = scores.index(max(scores))
        
        if max(scores) <= 1e-9: break
        
        selected_indices.append(idx)
        final_scores[idx] = scores[idx]
        
        # For all other candidates, compute distance to this one in feature space
        x_this = context["pool"][idx]["x"]
        for i in range(pool_size):
            if i not in selected_indices:
                dist_sq = np.sum((context['pool'][i]['x'] - x_this) ** 2)
                multiplier = 1.0 - np.exp(-dist_sq)
                scores[i] *= multiplier
                
    return final_scores