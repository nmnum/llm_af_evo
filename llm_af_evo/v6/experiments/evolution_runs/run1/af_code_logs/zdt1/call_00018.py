def modifier(context):
    """Suppress nearby duplicate candidates in objective space, promoting diverse exploration."""
    pool = context["pool"]
    y_obs = context["Y_obs"] 
    front_range = context["pareto_front_range"]
    
    # Normalize observations to [0, 1] for distance calculation  
    y_min = np.min(y_obs, axis=0)
    y_max = np.max(y_obs, axis=0) 
    
    # Avoid division by zero
    range_y = y_max - y_min 
    range_y[range_y == 0] = 1.0
    
    normalized_Y_obs = (y_obs - y_min) / range_y

    values = []
    
    for i in range(len(pool)):
        cand_x = pool[i]["x"]
        
        # Get candidate's predicted objectives
        gp_posterior = pool[i]["gp_posterior"] 
        pred_obj_vals = np.array([gp_posterior[name]['mean'] for name in context["objective_names"]])
                
        normalized_pred = (pred_obj_vals - y_min) / range_y
        
        min_dist_sq = float('inf')
        
        # Find minimum squared distance to any observed point
        for j, obs_point in enumerate(normalized_Y_obs):
            dist_sq = np.sum((normalized_pred - obs_point)**2)
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                
        # Apply suppression: the closer a candidate's prediction is 
        # to an observed point (in normalized objective space),  
        # the more we penalize it. Use squared distance for stronger penalty.
        
        values.append(-min(0.3, 15 * min_dist_sq)) 
    
    return values