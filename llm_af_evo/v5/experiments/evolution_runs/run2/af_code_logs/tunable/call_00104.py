def score_pool(context):
    """Balance acquisition value with progress-aware uncertainty and adaptive novelty to steer exploration toward under-covered regions."""
    names = context["objective_names"]
    
    # Normalize base scores by the observed range for consistency  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
        
    front_range = context["pareto_front_range"]

    unc_bonus_weight = 0.5
    nov_penalty_weight = 0.2

    # Uncertainty bonus: UCB-style, but scaled by how far we are from the ref point (progress-aware)
    campaign_progress = context['campaign']['progress']
    
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                             for name in names) 
        
        # Scale uncertainty bonus based on campaign progress — less exploration early, more later
        unc_bonus = (1.0 - campaign_progress) * 2.0 * sigma_norm_sum  
        
        unc_scores.append(unc_bonus_weight * unc_bonus)

    X_obs = context["X_obs"]
    
    nov_scores = []
    
    if len(X_obs) > 0:
        # Compute novelty as inverse distance to the nearest observed point, but only in objective space
        Y_obs = np.array([np.dot(cand['x'], [1.]*len(cand['x'])) for cand in context["pool"]])
        
        obs_dists_squared = []
    
        X_obs_obj_space = np.zeros((X_obs.shape[0], len(names)))
                
        # Assume features are ordered such that first d_features correspond to objective space
        if hasattr(context, 'n_objectives') and isinstance(context['n_objectives'], int):
            n_objs = context["n_objectives"]
            
            for i in range(len(X_obs)):
                X_obs_obj_space[i] = np.array([X_obs[i][j] 
                                               for j in range(n_objs)])
        else:
             # Fallback: assume all dimensions are objective space (not reliable)
             print("WARNING: n_objectives not specified, assuming features map to objectives.")
            
        
        if len(X_obs_obj_space) > 0 and Y_obs.shape[1:] == X_obs_obj_space.shape[1:]:
            for i in range(len(context["pool"])):
                x_cand = np.array([context['pool'][i]['x'][j] 
                                   for j in range(min(len(names),len(context['pool'][i]["x"])))])

                
                dists_to_observed_points_squared = (
                    (X_obs_obj_space[:, :len(x_cand)] - x_cand)**2).sum(axis=1)
                    
                min_dist_sq = np.min(dists_to_observed_points_squared) 
                obs_dists_squared.append(min_dist_sq)

            # Normalize inverse distances
            if len(obs_dists_squared) > 0:
                
                 max_inv_distance = 1. / (np.max(obs_dists_squared)**2 + 1e-8)
                 
                 nov_scores_scaled_by_range = [
                     -nov_penalty_weight * 
                      np.clip(1./((d+1e-6)**2), a_min=0, a_max=max_inv_distance)  
                    for d in obs_dists_squared]
            else:
                # Default to zero novelty penalty if no observations
                 nov_scores_scaled_by_range = [ 0. ] * len(context["pool"])
        else: 
             print("WARNING: Dimension mismatch or invalid feature/objective mapping.")
             nov_scores_scaled_by_range = np.zeros(len(context["pool"]))

    else:
         # No previous points observed — set to zero novelty penalties
          nov_scores_scaled_by_range = [ 0. ] * len(context["pool"])

    
     final_scores= []
        
     for i, _ in enumerate(context["pool"]):
         
        score = acq_scores[i] + unc_scores[i] - (nov_penalty_weight* np.mean(nov_scores_scaled_by_range)) 
        
        # Adjust weights dynamically to ensure sufficient exploration early on
        if campaign_progress < 0.3:
            weight_factor=1.
            
         elif campaign_progress >= 0.7: 
             score += (-unc_bonus_weight * unc_scores[i] + nov_penalty_weight* np.mean(nov_scores_scaled_by_range))
             
              # Reduce overall effect to focus more on exploitation
              
        
        final_scores.append(score)
        
    return final_scores