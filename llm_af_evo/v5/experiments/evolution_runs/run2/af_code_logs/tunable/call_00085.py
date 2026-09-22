def score_pool(context):
    """Blend acquisition value with a dynamic uncertainty-weighted novelty bonus that adapts based on campaign progress and front density."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute per-candidate distances to the nearest observed point
    X_obs = context["X_obs"] 
    scores = []
    for cand in context["pool"]:
        x_cand = cand["x"]
        
        if len(X_obs) == 0:
            dist_to_nearest = float('inf')
        else:
            # Euclidean distance from candidate to nearest observed point
            diffs = X_obs - x_cand 
            distances_squared = np.sum(diffs**2, axis=1)
            min_dist_sq = np.min(distances_squared)  
            if min_dist_sq == 0:   # Exact duplicate — treat as very far away for novelty purposes.
                dist_to_nearest = float('inf')
            else:
                dist_to_nearest = np.sqrt(min_dist_sq)

        acq_value_norm = cand["acq_value_norm"]
        
        # Compute uncertainty-normalized distance from the current front
        gp_posterior = cand["gp_posterior"] 
        pred_means = [gp_posterior[name]["mean"] for name in names]
        
        if len(context['pareto_front']) == 0:
            front_dist_to_nearest_sq = float('inf')
        else:  
            # Distance to the nearest point on pareto front
            diffs_pf = context["pareto_front"] - pred_means 
            distances_squared_pf = np.sum(diffs_pf**2, axis=1)
            min_dist_sq_pf = np.min(distances_squared_pf) 
            
            if min_dist_sq_pf == 0:
                # Predicted point lies exactly on the front
                dist_to_nearest_pf = float('inf')  
            else: 
                dist_to_nearest_pf = np.sqrt(min_dist_sq_pf)
                
        # Normalize uncertainty using range of objectives (for dimensionlessness)   
        sigma_norm_total = sum(gp_posterior[name]["std"] / front_range[name] for name in names)

        if context["campaign"]["progress"] < 0.3:
            # Early phase: emphasize exploration via novelty bonus
            weight_uncertainty = 1e-2  
            base_weight_novelty = 5.
            
        elif context['campaign']['stagnant_batches'] >= 4 and len(context['pareto_front']) > 1 :
             # Stuck in stagnation? add a strong uncertainty + front proximity bonus to break out
            weight_uncertainty = .2  
            base_weight_novelty = 3. 
        else:
            # Later phase: moderate balance between exploitation (acq), exploration via novelty, and robustness.
            weight_uncertainty = .05   
            base_weight_novelty = 1.

        
         if dist_to_nearest == float('inf'):
             novel_score_cand = np.nan
         elif len(context["Y_obs"]) > 2:
              # Use inverse of distance (so far away means high score) and scale by front range.
              normalised_distance_inv = min(50., max(.1, dist_to_nearest / .3))  
              novel_score_cand = np.log(normalised_distance_inv)
         else: 
             novel_score_cand = 2. * (dist_to_nearest > 4.)

        # Final score is a blend of acquisition value with dynamic uncertainty and novelty terms
        final_score = acq_value_norm + \
                      weight_uncertainty * sigma_norm_total -\
                      base_weight_novelty * np.nanmean([novel_score_cand]) 

        
    return scores