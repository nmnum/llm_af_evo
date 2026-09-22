def modifier(context):
    """Penalize candidates close to already-selected ones in feature space, reducing batch duplication."""
    pool = context["pool"]
    values = [0.0] * len(pool)
    
    # Collect selected points from previous batches (if any) - simplified assumption 
    # that we are working with a single batch selection and not tracking full history
    if hasattr(modifier, 'selected_x'):
        selected_x = modifier.selected_x  
    else:
        selected_x = []
        
    for i in range(len(pool)):
        cand_x = pool[i]["x"]
        
        min_dist_sq = float('inf')
        # Find minimum squared distance to any previously selected candidate
        for prev_x in selected_x:
            dist_sq = np.sum((cand_x - prev_x) ** 2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        
        # Scale penalty based on how close the point is (0.3 max penalization)
        distance = np.sqrt(min_dist_sq) if min_dist_sq != float('inf') else 1e-6
        values[i] = -min(0.3, 2 * distance)

    modifier.selected_x = selected_x + [cand_x]
    
    return values