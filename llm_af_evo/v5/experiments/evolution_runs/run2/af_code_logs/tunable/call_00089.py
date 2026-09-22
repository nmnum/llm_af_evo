def score_pool(context):
    """Score candidates by base acquisition value, then iteratively reduce scores of nearby picks to encourage diversity."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Initialize all scores with acq_value_norm (base score)
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    final_scores = base_scores.copy()
    
    picked_indices = []
    current_pool_mask = np.ones(pool_size, dtype=bool)

    while len(picked_indices) < pool_size:
        # Pick the best candidate among unpicked ones
        if not np.any(current_pool_mask):
            break
        
        top_idx = np.argmax(final_scores[current_pool_mask]) + (len(final_scores) - np.sum(current_pool_mask))
        
        picked_indices.append(top_idx)
        current_pool_mask[top_idx] = False

        # Compute distances from this candidate to all others
        x_this = context["pool"][top_idx]["x"]
        dists_x = [np.linalg.norm(x_this - cand['x']) for i, cand in enumerate(context["pool"]) if i != top_idx]
        
        # Update scores of remaining candidates using multiplier based on distance from picked candidate(s)
        new_multipliers = []
        for j, _ in enumerate(final_scores):
            if j == top_idx:
                continue
            
            dist_x_to_j = np.linalg.norm(x_this - context["pool"][j]["x"])
            
            multiplier = 1.0 - np.exp(-dist_x_to_j)  
            new_multipliers.append(multiplier)
        
        # Apply multipliers to scores of remaining candidates
        for j, mult in enumerate(new_multipliers):
            if current_pool_mask[j]:
                final_scores[j] *= max(0., mult)

    return list(final_scores.tolist())