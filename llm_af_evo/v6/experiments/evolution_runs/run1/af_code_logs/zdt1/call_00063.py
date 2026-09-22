def modifier(context):
    """Penalize candidates that are too close to previously-ranked higher-scoring ones, discouraging batch duplication."""
    values = []
    selected_x = []  # Store x of already-selected (higher-rank) points
    small_scale = 0.3
    
    # Sort indices by acq_value_norm descending (highest first)
    sorted_indices = [i for i, _ in sorted(enumerate(context["pool"]), key=lambda x: -x[1]["acq_value_norm"])]
    
    for idx in sorted_indices:
        cand_x = context["pool"][idx]["x"]
        
        # Compute minimum distance to all selected points so far
        min_dist_sq = float('inf')
        for prev_x in selected_x:
            dist_sq = np.sum((cand_x - prev_x) ** 2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        
        # If no previous point, or very distant, multiplier is near 1.0
        if min_dist_sq == float('inf') or min_dist_sq > 5.0: 
            multiplier = 1.0  
        else:
            # Use a decreasing exponential function of distance to determine penalty scaling factor (in [0., 1])
            multiplier = 1.0 - np.exp(-min_dist_sq)
        
        values.append((multiplier - 1.0) * small_scale)
        selected_x.append(cand_x)

    # Return correction terms in original pool order
    result = [values[sorted_indices.index(i)] for i in range(len(context["pool"]))]
    
    return result