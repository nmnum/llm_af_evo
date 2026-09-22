def score_pool(context):
    """Score candidates by base acquisition value, then greedily select diverse batch by reducing scores of nearby contenders."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Start with acq_value_norm as the base score for each candidate
    base_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    selected_indices = []
    final_scores = [0.0] * pool_size
    
    # Greedily pick candidates, reducing scores of nearby ones
    picked_x = []  # Store x positions that have been chosen

    while len(selected_indices) < min(5, pool_size):   # Assuming batch size is around 5 or less; adjust as needed.
        if not any(s > 0 for s in base_scores):
            break
        
        best_idx = -1
        max_score = -float('inf')
        
        for i, score in enumerate(base_scores):
            if score <= 0:
                continue
            
            # If candidate is already selected (shouldn't happen), skip it.
            if i in selected_indices:
                continue
                
            if score > max_score:
                best_idx = i
                max_score = score

        if best_idx == -1: 
            break
        
        final_scores[best_idx] = base_scores[best_idx]
        
        # Record the chosen candidate's x position for future comparisons.
        selected_indices.append(best_idx)
        picked_x.append(context["pool"][best_idx]["x"])

        # Update scores of remaining candidates based on proximity to already-picked ones
        for i, cand in enumerate(context["pool"]):
            if i in selected_indices:
                continue
            
            min_dist = float('inf')
            
            x_cand = cand["x"]
                
            for picked_x_pos in picked_x:  # Compare against all previously chosen points.
                dist_sq = np.sum((picked_x_pos - x_cand) ** 2)
                if dist_sq < min_dist:
                    min_dist = dist_sq

            multiplier = 1.0 - np.exp(-min_dist)

            base_scores[i] *= max(0., multiplier)

    return final_scores