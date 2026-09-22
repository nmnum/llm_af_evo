def score_pool(context):
    """Rank candidates greedily by base acquisition value, then reduce scores of similar remaining candidates based on distance to already-picked ones."""
    names = context["objective_names"]
    
    # Base scoring using normalized acq_value_norm (already computed hypervolume improvement)
    base_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    picked_indices = []
    final_scores = [-float('inf')] * len(context["pool"])
    
    while len(picked_indices) < len(context["pool"]):
        # Find best unpicked candidate by base score
        max_idx = -1
        max_base_score = -float('inf')
        
        for i in range(len(base_scores)):
            if i not in picked_indices and base_scores[i] > max_base_score:
                max_base_score = base_scores[i]
                max_idx = i
                
        # If no unpicked candidates, break (shouldn't happen)
        if max_idx == -1: 
            break
            
        picked_indices.append(max_idx) 
        
        # Compute multiplier for all remaining candidates based on distance to this one
        cand_x = context["pool"][max_idx]["x"]
        
        multipliers = []
        for i in range(len(context["pool"])):
            if i not in picked_indices:
                other_cand = context["pool"][i]
                
                # Distance in x-space (L2 norm)
                dist_x_sq = np.sum((cand_x - other_cand['x']) ** 2) 
                
                multiplier = 1.0 - np.exp(-dist_x_sq)
                multipliers.append(multiplier)
            else:
                multipliers.append(1.0)

        # Assign final score (base * multiplier), store for next iteration
        base_score_for_this_pick = max_base_score
        
        if len(picked_indices) == 1: 
            # First pick, no reduction yet; use raw scores  
            current_scores = [s * m for s,m in zip(base_scores, multipliers)]
            
        else:
            # Reuse previous final score (already multiplied by its own multiplier)
            prev_score_for_this_pick = base_score_for_this_pick
            updated_current_scores = []
        
            for i,(base_s, mult) in enumerate(zip(current_scores_before_mult, multipliers)):
                if i not in picked_indices: 
                    new_val = base_s * mult  # This is the correct final score formula  
                    updated_current_scores.append(new_val)
                    
        current_scores_before_mult = [s*mult for s,mult in zip(base_scores, multipliers)]
        
    return list(current_scores_before_mult)