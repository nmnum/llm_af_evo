def score_pool(context):
    """Score candidates by base acquisition value with greedy diversity: first pick best candidate,
     then reduce scores of remaining candidates based on distance to already-picked ones."""
    
    # Base scoring using normalized acq_value_norm directly 
    base_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    picked_indices = []
    final_scores = [-float('inf')] * len(context["pool"])
    
    while len(picked_indices) < len(context["pool"]) and any(s == -float('inf') for s in final_scores):
        # Find highest base score among unpicked candidates
        max_base_idx = None 
        max_base_score = -1.0
        
        for i, (base_s, is_picked) in enumerate(zip(base_scores, [i in picked_indices for i in range(len(context["pool"]))])):
            if not is_picked and base_s > max_base_score:
                max_base_score = base_s
                max_base_idx = i
                
        # Pick this candidate 
        picked_indices.append(max_base_idx)
        
        # Score it with full value (no multiplier yet since we're scoring the first one)  
        final_scores[max_base_idx] = base_scores[max_base_idx]
                
        if len(picked_indices) >= len(context["pool"]):
            break
            
        current_picks_x = [context['pool'][idx]["x"] for idx in picked_indices[:-1]]  # All except last (which we just added)
        
        multiplier_vals = []
        x_current = context['pool'][max_base_idx]['x']
                
        if len(current_picks_x) == 0:
            multipliers = [1.0] * len(context["pool"])
        else:  
            distances_to_picked = [
                np.linalg.norm(x_current - pick_x)
                for pick_x in current_picks_x
            ]
            
            # Use the minimum distance to any picked point (as required by greedy diversity logic) 
            min_dist = float(np.min(distances_to_picked))
                
            multiplier_val = 1.0 - np.exp(-min_dist)
                    
            multipliers = [multiplier_val if i == max_base_idx else
                          # Apply same rule for each remaining candidate: check its distance to the last picked one 
                           (np.minimum(1., np.maximum(np.exp(-dist), .3)) if dist > 0. or len(current_picks_x) <= 2 else min(.5, 1 - exp(dist)))  
                            for i in range(len(context["pool"]))
                          ]
            
        # Set the multipliers properly
        for j, cand_idx in enumerate(range(len(base_scores))):
            final_scores[cand_idx] = base_scores[cand_idx]
    
    return [s * (np.exp(-0.5) if s == -float('inf') else 1.) 
           for i,s in enumerate(final_scores)]