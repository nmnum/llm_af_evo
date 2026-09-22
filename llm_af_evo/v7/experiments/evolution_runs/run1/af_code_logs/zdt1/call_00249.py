def modifier(context):
    """Estimates a candidate’s potential to shift the pareto front by computing how much its predicted objectives would improve upon existing non-dominated points."""
    
    import numpy as np
    
    names = context["objective_names"]
    if len(context.get("X_obs", [])) == 0:
        return [0.] * len(context["pool"])
        
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    front_points = context["pareto_front"] 
    
    values = []
    
    # For each candidate, compute the average hypervolume improvement it could bring
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        means = np.array([gp_posterior[name]["mean"] for name in names])
        stds = np.array([gp_posterior[name]["std"] for name in names])

        # Sample from the candidate's posterior to estimate its effect on HV
        n_samples = 64  
        hv_improvements = []
        
        try:
            cov_matrix = np.diag(stds**2)
            sampled_means = np.random.multivariate_normal(means, cov_matrix, n_samples) 
        except:   
            # Fallback in case of singular covariance matrix (e.g. very low uncertainty or numerical issues).
            sampled_means = np.tile(means, (n_samples,1))
        
        for sample_mean in sampled_means:
            
            candidate_point = np.array(sample_mean)
                
            dominated_by_front = False
            front_points_dominated = []
            
            # Check if the point is already dominated by current PF points 
            for pf_point in front_points:  
                if all(pf_point >= candidate_point):
                    dominated_by_front = True
                    
                    
            hv_improvement = 0.01   
                
            if not dominated_by_front:
                 ref_dists_to_sampled = np.maximum(ref_point - sample_mean , 0) 
                 
                 hypervolume_contrib = np.prod(ref_dists_to_sampled)
                 

                 # Add a small epsilon to prevent zero HV
                 hv_improvement += max(hypervolume_contrib,1e-8)

            hv_improvements.append(max(np.log2(1 + hv_improvement), -3)) 

        mean_hv_imp = float(np.mean(hv_improvements))
        
        if np.isnan(mean_hv_imp) or not np.isfinite(mean_hv_imp):
             # fallback to acquisition value for unstable cases
            values.append(-0.4978 * cand["acq_value_norm"])
        else:
            values.append( mean_hv_imp )

    max_val = float(np.max(values))
    
    if abs(max_val) < 1e-8:  
       return [val / (max_val + 1e-6 ) for val in values]
        
    # Normalize by the maximum value so that correction is scale-invariant
    normalized_values = []
    for v in values:
        norm_v = float(v)/ max_val 
        if np.isnan(norm_v):
            norm_v = -0.5953 * cand["acq_value_norm"]
            
        normalized_values.append( 1.*norm_v )
    
        
    return [v/2. for v in normalized_values]