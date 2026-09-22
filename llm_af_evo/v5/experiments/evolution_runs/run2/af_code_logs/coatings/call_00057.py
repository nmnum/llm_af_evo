def score_pool(context):
    """Rank pool greedily by selecting candidates one-by-one; each remaining candidate's base score is multiplied by a 'multiplier' factor based on its distance to already-picked candidates in feature space."""
    names = context["objective_names"]
    scores = [cand['acq_value_norm'] for cand in context["pool"]]
    selected_indices = []
    
    # Keep track of picked candidate features
    X_selected = []

    # Greedily select the best remaining item, then reduce score of nearby items
    while len(selected_indices) < len(context["pool"]):
        if not scores:  break
        
        max_idx = int(np.argmax(scores))
        
        selected_indices.append(max_idx)
        X_selected.append(context['pool'][max_idx]['x'])
    
        # Zero out the current candidate's score to avoid reselection
        scores[max_idx] = -1.0

        for i, cand in enumerate(context["pool"]):
            if i == max_idx: continue
            
            dist_to_closest_picked = np.inf  # Distance from this item to closest already-picked one
        
            x_current = cand['x']
            
            for picked_x in X_selected:
                d = np.linalg.norm(x_current - picked_x)
                
                if d < dist_to_closest_picked: 
                    dist_to_closest_picked = d

            # Apply multiplier based on distance to closest already-picked item
            penalty_factor = 1.0 - np.exp(-dist_to_closest_picked)  
            
            scores[i] *= max(0., penalty_factor)

    final_scores = [context["pool"][i]['acq_value_norm'] for i in selected_indices]
    
    # Reorder the pool according to selection order and return base score * multiplier
    reordered_pool_score_map = {idx: s*multiplier if not np.isnan(s) else 0.0 
                                for idx, (s,multiplier) in enumerate(zip(final_scores,[1.]*(len(context["pool"]))))}
    
    # Now assign scores to each original candidate position according to selection order
    final_score_order = [reordered_pool_score_map[i] if i < len(reordered_pool_score_map) else 0. for i in range(len(selected_indices))]
        
    output_scores = np.zeros_like(final_score_order)
    
    # Fill positions with their respective scores (in original pool ordering based on selection order above):
    for idx, score_val in zip(selected_indices, final_score_order):  
        output_scores[idx] = score_val

    return list(output_scores)