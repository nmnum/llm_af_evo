def score_pool(context):
    """Blend acquisition value with uncertainty-weighted progress toward Pareto front boundaries and novelty."""
    names = context["objective_names"]
    
    # Use normalized acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute distance to nearest point on the current pareto front
    X_obs = context["X_obs"]
    pf_points = context["pareto_front"]

    if len(pf_points) == 0:
        novelty_scores = np.zeros(len(context["pool"]))
    else:  
        cand_x = np.array([cand['x'] for cand in context["pool"]])
        
        # Compute min distance from each candidate to any point on the Pareto front
        distances_to_pf = []
        pf_points_np = np.array(pf_points)
            
        for i, x_cand in enumerate(cand_x):
            dists = [np.linalg.norm(x_cand - pt) for pt in pf_points_np]
            min_dist = min(dists) if len(dists) > 0 else float('inf')
            distances_to_pf.append(min_dist)

        # Normalize novelty scores to [0,1] range
        max_distance = np.max(distances_to_pf)
        
        if max_distance == 0:
            novel_scores_normalized = np.zeros(len(context["pool"]))
        else:  
            novelty_scores = (max_distance - np.array(distances_to_pf)) / max_distance
            
    # Use uncertainty-based progress metric to encourage exploration near frontiers
    sigma_weights = []
    
    for cand in context["pool"]:
        gp_posterior = cand['gp_posterior']
        
        total_sigma_weighted = 0.0
        
        if len(pf_points) > 0:
            pf_range_per_obj = [context["pareto_front_range"][name] 
                                for name in names]
            
            # For each objective, compute how far mean is from the front boundary
            obj_means = np.array([gp_posterior[name]["mean"] for name in names])
                
            if len(pf_points) > 0:
                pf_min_vals_per_obj = [min(pt[i] for pt in pf_points)
                                       for i in range(len(names))]
                    
                # Compute progress toward the front (how much improvement is possible per obj,
                # weighted by inverse of uncertainty to focus on informative directions).
                
                for j, name in enumerate(names):
                    mu_j = gp_posterior[name]["mean"]
                        
                    if pf_range_per_obj[j] > 0:
                        dist_to_front_boundary_normed = abs(mu_j - min(pf_min_vals_per_obj)) / \
                                                       (pf_range_per_obj[j])
                    
                        # Add a small amount of uncertainty-weighting to guide exploration
                        sigma_j_inv = max(1e-6, gp_posterior[name]["std"])
                        
                    else:
                        dist_to_front_boundary_normed = 0.5
                        
                        # Small fixed inverse weight if no range yet defined.
                        sigma_j_inv = 1
                    
                total_sigma_weighted += (dist_to_front_boundary_normed / 
                                         max(1e-6, gp_posterior[name]["std"]))

        else:
            for name in names:  
                 total_sigma_weighted += 1.0
            
        # Weight by number of objectives to normalize
        sigma_weights.append(total_sigma_weighted)

    combined_scores = []
    
    alpha_acq = .7 
    beta_novelty = .25
    
    if len(pf_points) > 0:
         gamma_progress = 0.3  
     else:   
          # Early on, prioritize acquisition and novelty over progress-based scores
          gamma_progress = 0.
          
    for i in range(len(acq_scores)):
        final_score = (alpha_acq * acq_scores[i] + 
                       beta_novelty * np.clip(novelty_scores[i], 0.,1.) +
                        gamma_progress * sigma_weights[i])
                        
        combined_scores.append(final_score)
        
    return combined_scores