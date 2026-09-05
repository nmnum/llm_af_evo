def score_pool(context):
    """Blend acquisition value with progress-aware uncertainty and novelty to balance exploration-exploitation."""
    names = context["objective_names"]
    ref_point_by_name = context['ref_point_by_name']
    
    # Normalize acq_value_norm from botorch's qLogNEHVI 
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Compute uncertainty as normalized std deviation across objectives
        front_range = np.array([ref_point_by_name[name] for name in names])
        sigma_norm = sum(gp[name]["std"] / (front_range[i]+1e-9) 
                         for i, name in enumerate(names))
        
        acq_value = cand["acq_value_norm"]
        
        # Use progress-aware weights: early=more uncertainty+novelty; late=exploit more
        step_ratio = context['campaign']['progress']
        w_exploit = 0.3 + 0.7 * (1 - step_ratio) 
        w_uncertainty = 2.5 * step_ratio  
        
        # Add novelty term: distance to nearest observed point in feature space, normalized by total range
        X_obs = context["X_obs"]
        if len(X_obs):
            x_cand = cand['x']
            distances_sq = np.sum((X_obs - x_cand) ** 2, axis=1)
            min_dist = np.sqrt(np.min(distances_sq))
            
            # Normalize by total feature space range (assumed to be unit cube for each dimension after normalization)
            novelty_score = min_dist 
        else:
            novelty_score = 0.5
            
        final_score = w_exploit * mu_sum + \
                      w_uncertainty * sigma_norm + \
                      acq_value - \ # subtract because acquisition is already a gain estimate
                      (1e-3) * novelty_score
        
        scores.append(final_score)
    
    return scores