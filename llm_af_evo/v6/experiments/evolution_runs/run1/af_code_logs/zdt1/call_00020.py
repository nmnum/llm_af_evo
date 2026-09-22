def modifier(context):
    """Diversity bonus based on repelling candidates that are close in feature space to already selected ones."""
    pool = context["pool"]
    values = [0.0] * len(pool)
    
    # If no observations yet, return flat correction
    if not context["X_obs"].size:
        return values

    # Compute squared distances from each candidate's x to all previously observed points  
    selected_x = []
    for i in range(len(pool)):
        cand_x = pool[i]["x"]
        
        min_dist_sq = float('inf')
        for prev_x in selected_x:
            dist_sq = np.sum((cand_x - prev_x) ** 2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        
        # Apply repulsion: higher bonus when farther from already-selected points
        distance = np.sqrt(min_dist_sq) if min_dist_sq != float('inf') else 0.1  
        
        values[i] = max(0., (distance - 0.2)) * 0.3

        selected_x.append(cand_x)

    return values