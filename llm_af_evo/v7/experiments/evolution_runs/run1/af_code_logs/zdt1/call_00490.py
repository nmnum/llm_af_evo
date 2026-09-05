def modifier(context):
    """Penalize candidates that are close in feature space to already-selected top candidates, encouraging diverse batch selection."""
    pool = context["pool"]
    if len(pool) == 0:
        return []
    
    # Collect selected candidate features from previous batches (if any)
    try:
        X_selected = np.array([cand['x'] for cand in pool[:-len(context.get('batch', [])) or -1]])
    except Exception:
        # Fallback: no batch history available, treat as empty selection
        X_selected = []
    
    values = [0.0] * len(pool)
    selected_indices = []

    for i, cand in enumerate(pool):
        x_i = cand["x"]
        
        if not isinstance(x_i, np.ndarray) or (len(X_selected) > 0 and 
                                               all(np.allclose(x_i, prev_x, atol=1e-6) for prev_x in X_selected)):
            # Skip already-selected candidates to avoid double penalization
            values[i] = -np.inf  
        else:
            min_dist_sq = float('inf')
            
            if len(X_selected) > 0 and not np.allclose(x_i, x_i): 
                continue
                
            for j in range(len(selected_indices)):
                prev_x = X_selected[selected_indices[j]]
                
                # Compute squared Euclidean distance
                dist_sq = sum((x - px)**2 for (x, px) in zip(x_i, prev_x))
            
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    
            penalty_factor = 0.3 * np.exp(-min_dist_sq)
                
        values[i] += (-penalty_factor)

    return values