def modifier(context):
    """Penalize candidates in pool that are close to higher-ranked already-selected items, using a multiplier based on Euclidean distance."""
    batch_size = len(context["pool"])
    selected_x = []
    values = [0.0] * batch_size
    
    for i in range(batch_size):
        cand_x = context["pool"][i]["x"]
        
        # Compute minimum distance to already-selected candidates
        min_dist_sq = float('inf')
        for prev_x in selected_x:
            dist_sq = np.sum((cand_x - prev_x) ** 2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        
        # If no previous candidate, multiplier is high (1.0), otherwise low
        if len(selected_x) > 0 and min_dist_sq < float('inf'):
            distance = np.sqrt(min_dist_sq)
            multiplier = 1.0 - np.exp(-distance)
        else:
            multiplier = 1.0
            
        # Apply correction: negative value to penalize clustering, scale it
        values[i] = (multiplier - 1.0) * 0.3
        
        selected_x.append(cand_x)

    return values