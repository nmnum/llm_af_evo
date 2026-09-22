def modifier(context):
    """Penalize candidates in pool that are close to higher-ranked ones already considered, using a multiplicative factor based on Euclidean distance."""
    if len(context["pool"]) <= 1:
        return [0.0] * len(context["pool"])
    
    # Start with the acq_value_norm scores for ranking
    ranked_indices = sorted(range(len(context["pool"])), key=lambda i: context["pool"][i]["acq_value_norm"], reverse=True)
    
    used_x_set = set()
    values = [0.0] * len(context["pool"])
    small_scale = 0.3
    
    for idx in ranked_indices:
        cand_x = tuple(context["pool"][idx]["x"]) 
        multiplier = 1.0
        
        # Check distance to all previously considered candidates
        if used_x_set:  
            min_dist_sq = float('inf')
            
            for prev_cand_x in used_x_set:
                dist_sq = sum((a - b) ** 2 for a, b in zip(cand_x, prev_cand_x))
                
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    
            # Use the distance to compute multiplier
            penalty_factor = small_scale * (1.0 - np.exp(-min_dist_sq)) 
            multiplier -= penalty_factor
            
        used_x_set.add(cand_x)
        
    return [values[i] for i in range(len(context["pool"]))]