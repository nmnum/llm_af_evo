def score_pool(context):
    """Use acquisition value scaled by the expected improvement ratio and penalize candidates near stagnant progress regions."""
    names = context["objective_names"]
    
    # Get base scores from normalized acquisition values  
    acq_scores = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    
    # Compute uncertainty bonus with dynamic weighting
    progress = context["campaign"]["progress"]
    ucb_bonus_weight = 0.5 * (1 - progress)
    front_range = context["pareto_front_range"]
    unc_scores = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                             for name in names)
        unc_scores.append(ucb_bonus_weight * sigma_norm_sum)

    # Identify stagnant regions using recent observations
    X_obs = context["X_obs"]
    Y_obs = context["Y_obs"]
    
    nov_penalty_weight = 0.3
    
    if len(X_obs) > 1:
        obs_dists = []
        
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            
            min_dist = np.min(dists)

            # Normalize by the number of features  
            normalized_min_dist = min_dist / len(x_cand) 
            
            obs_dists.append(normalized_min_dist)
        
        nov_scores = [-nov_penalty_weight * d for d in obs_dists]
    else:
        nov_scores = [0.0] * len(context["pool"])
    
    # Compute expected improvement ratio to scale acquisition value
    if progress < 0.5 and context['campaign']['stagnant_batches'] > 2:  
        
        front_size = Y_obs.shape[0]

        ref_point_scaled = np.array([context["ref_point_by_name"][name] 
                                    for name in names])

        # Compute hypervolume of current Pareto front
        hv_front = compute_hypervolume(Y_obs, ref_point_scaled)

        if hv_front > 1e-8:
            # Use the ratio between max possible HV and actual to adjust acquisition scores  
            total_volume_possible = np.prod(ref_point_scaled)
            
            improvement_ratio = (total_volume_possible - hv_front) / total_volume_possible
        else: 
            improvement_ratio = 0.5
            
    elif progress >= 0.7 or context['campaign']['stagnant_batches'] < 2:
        
        # In later stages, scale with the inverse of front size to encourage diversity  
        if front_size > 1:
            
            improvement_ratio = max(0., (front_size - 3) / (front_size + 5))
        else: 
          
            improvement_ratio = 0.5
            
    else:

        # Mid-stage, scale slightly down
        improvement_ratio = progress * 2  

    
    final_scores = acq_scores * improvement_ratio \
                   + np.array(unc_scores)\
                   + np.array(nov_scores)
        
    return list(final_scores)

def compute_hypervolume(front_points, ref_point):
   
    try:
       
        from scipy.spatial import ConvexHull
        

      
        hull = ConvexHull(front_points)  
 
     
        volume = 0.0
        
    
        for simplex in hull.simplices: 
            
            vertices = front_points[simplex]
            
          
            if len(vertices.shape) == 1:
                continue
                
           
            diffs = ref_point - np.array([v.min(axis=0) for v in [vertices]])
        
      
            volume += abs(np.prod(diffs))
       
        return min(volume, 2.5e6)
    
    except Exception:

     
       # Fallback: use a rough estimate 
   
      try:
          front_min = np.min(front_points,axis=0)
          
          diffs = ref_point - front_min
         
          volume_estimate = abs(np.prod(diffs))
        
      
          return min(volume_estimate, 2.5e6) 
        
  
      except Exception:

       
        # Last resort: just a fixed small value 
   
       return float(1.)