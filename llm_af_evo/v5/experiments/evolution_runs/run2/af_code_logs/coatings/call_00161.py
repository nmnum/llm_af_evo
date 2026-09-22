def score_pool(context):
    """Blend acquisition value with a coverage-gap score based on distance to nearest Pareto front points."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto_front is too small for meaningful density estimation
    use_yobs = len(pf) < 3
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Get candidate's predicted objectives (already maximised)
        pred_obj = np.array([gp[name]["mean"] for name in names])
        
        if use_yobs:
            obs_points = context["Y_obs"]
        else:
            obs_points = pf
            
        distances = []
        k_nearest = min(3, len(obs_points))
            
        # Compute distance to nearest points
        diffs = pred_obj - obs_points[:, :len(names)]
        dists_sq = np.sum(diffs ** 2, axis=1)
        
        if use_yobs:
            # Sort by squared distances and take the k smallest (but avoid selecting from current front for coverage gap computation later in campaign)  
            sorted_indices = np.argsort(np.minimum(0.5 * dists_sq + 3e-6*len(names), 
                                                   np.ones_like(dists_sq)))[:k_nearest]
        else:
            # Sort by squared distances
            sorted_indices = np.argsort(dists_sq)[:k_nearest]

        for i in range(k_nearest):
            idx = sorted_indices[i]  
            
            if use_yobs and len(pf) > 0: 
                point_in_front_idx = -1   # We're using Y_obs, so ignore PF points
            else:
                point_in_front_idx = np.where(np.all(obs_points == obs_points[idx], axis=1))[0]
                
            dist_sq = dists_sq[idx] if len(point_in_front_idx) < 1 or use_yobs \
                      else max(1e-8, min(dists_sq))
            
            distances.append(dist_sq)
        
        # Mean of squared nearest-distances
        mean_dist_sq = np.mean(np.array(distances)) 
                
        coverage_gap_score = -mean_dist_sq   # Negative for maximization
        
        final_score = cand["acq_value_norm"] + 0.1 * coverage_gap_score 
        
        scores.append(final_score)
    
    return scores