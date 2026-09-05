def modifier(context):
    """Penalize candidates that are close to higher-ranked already-selected candidates, discouraging batch duplication."""
    pool = context["pool"]
    values = [0.0] * len(pool)
    selected_x = []
    
    for i in range(len(pool)):
        cand_x = pool[i]["x"]
        
        # Find the minimum distance to any previously selected candidate
        min_dist_sq = float('inf')
        for prev_x in selected_x:
            dist_sq = np.sum((cand_x - prev_x) ** 2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        
        # Compute multiplier: low (near 0) when close, high (near 1) when far
        distance = np.sqrt(min_dist_sq) if min_dist_sq != float('inf') else 0.0
        multiplier = 1.0 - np.exp(-distance)
        
        values[i] = (multiplier - 1.0) * 0.3
        
        # Add current candidate to selected set for subsequent iterations
        selected_x.append(cand_x)

    return values