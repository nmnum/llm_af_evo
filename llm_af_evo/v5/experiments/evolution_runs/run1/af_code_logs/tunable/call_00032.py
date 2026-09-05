def score_pool(context):
    """Blend acquisition value with progress-aware uncertainty and novelty to balance exploration-exploitation."""
    
    names = context["objective_names"]
    ref_point_by_name = context['ref_point_by_name']
    X_obs = context['X_obs'] 
    front_range = context["pareto_front_range"]

    # Compute scores based on acq_value_norm, normalized uncertainty (scaled by range), and novelty
    scores = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Use the provided acquisition value directly 
        acqu_val = cand['acq_value_norm']
        
        # Compute scaled standard deviation across objectives  
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        
        # Normalize candidate's x to [0, 1]^d space
        norm_x = np.array(cand["x"])
                
        # Calculate novelty as inverse of distance to nearest observed point 
        if len(X_obs) > 0:
            distances_sq = np.sum((X_obs - norm_x)**2, axis=1)
            min_dist_squared = np.min(distances_sq)
            
            # Avoid division by zero
            novel_score = 1. / (min_dist_squared + 1e-9)
        else: 
            novel_score = 0.
        
        # Combine components with weights that change based on campaign progress  
        p = context["campaign"]["progress"]
                
        exploit_weight   = np.clip(2 * p, 0., 1.)      # Increase exploitation as we go
        explore_weight   = max(0.5 - p, 0)             # Decrease exploration after mid-point 
        novelty_weight   = min(p + 0.3, 1.)

        score = (exploit_weight * acqu_val +
                 explore_weight * sigma_sum +
                 novelty_weight * novel_score)
        
        scores.append(score)

    return scores