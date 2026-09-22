def score_pool(context):
    """Rank candidates greedily by base acquisition value, then suppress scores of nearby unpicked candidates using an exponential decay multiplier based on distance in feature space."""
    if not context["pool"]:
        return []
    
    names = context["objective_names"]
    pool_size = len(context["pool"])
    picked_indices = set()
    final_scores = [0.0] * pool_size
    base_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Greedily select candidates and suppress nearby ones
    while len(picked_indices) < pool_size:
        best_idx = -1
        highest_score = float('-inf')
        
        # Find the unpicked candidate with maximum base score
        for i, (is_picked, base_scor) in enumerate(zip(picked_indices, base_scores)):
            if not is_picked and base_scor > highest_score:
                best_idx = i
                highest_score = base_scor
                
        picked_indices.add(best_idx)
        
        # Assign final score to the selected candidate as its full base value (no multiplier applied yet here)
        final_scores[best_idx] = highest_score
        
    # Now compute multipliers for unpicked candidates based on distance from already-picked ones
    all_picked_x = [context["pool"][i]["x"] for i in picked_indices]
    
    # For each remaining candidate, calculate multiplier as 1 - exp(-distance)
    scores_with_multiplier = []
    for idx, cand in enumerate(context["pool"]):
        if idx not in picked_indices:
            min_dist_sq = float('inf')
            
            x_cand = cand['x']
            for x_picked in all_picked_x:
                dist_sq = np.sum((x_cand - x_picked) ** 2)
                
                # Take minimum squared distance
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    
            multiplier = max(0.0, 1.0 - np.exp(-min_dist_sq))
        else:
            multiplier = 1.0
            
        scores_with_multiplier.append(base_scores[idx] * multiplier)
    
    return scores_with_multiplier