def modifier(context):
    """Resamples posterior means with added noise to assess potential for dominating existing points and rewards candidates that could shift the Pareto front."""
    
    import numpy as np
    
    names = context["objective_names"]
    if len(context.get("X_obs", [])) == 0:
        return [0.] * len(context["pool"])
        
    # Use a fixed reference point to compute hypervolume contributions
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    values = []
    n_samples = 128
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        means = [gp_posterior[name]["mean"] for name in names]
        stds = [gp_posterior[name]["std"] for name in names]

        # Sample noisy versions of the candidate's mean
        try:
            cov_matrix = np.diag(np.array(stds)**2)
            sampled_means = np.random.multivariate_normal(means, cov_matrix, n_samples) 
        except:  
            sampled_means = np.tile(means, (n_samples,1))
        
        # Count how many noisy samples would dominate or be dominated by current front
        dominates_count = 0.0
        
        for sample in sampled_means:
            
            candidate_point = np.array(sample)
                
            is_dominated_by_front = False
            
            for pf_point in context["pareto_front"]:  
                if all(pf_point >= candidate_point):
                    is_dominated_by_front = True
                    break
                    
                    
            # If not dominated by front, check how much it could expand HV (but also consider dominance)
            
            hv_contrib = 0.0   
                
            if not is_dominated_by_front:
                 ref_dist_to_sampled = np.maximum(ref_point - sample , 0) 
                 
                 hypervolume_contrib = np.prod(ref_dist_to_sampled)

                 # Add a small epsilon to prevent zero HV
                 hv_contrib += max(hypervolume_contrib,1e-8)
            
            if not is_dominated_by_front:
                dominates_count += 1.0

        dominance_ratio = float(dominates_count) / n_samples
        
        values.append(2 * (dominance_ratio - 0.5)) # Center around zero; scale to make impact visible
    
    max_val = np.max(values)
    
    if abs(max_val) < 1e-8:  
       return [val for val in values]
        
    normalized_values = []
    for v in values:
        norm_v = float(v)/ (max_val + 1e-6 ) 
        normalized_values.append(0.5 * norm_v)
    
    # Normalize so that correction is scale-invariant and moderate
    return [v/2. for v in normalized_values]