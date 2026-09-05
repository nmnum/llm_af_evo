def score_pool(context):
    """Resamples candidate objectives under Gaussian noise and estimates expected hypervolume gain."""
    import numpy as np
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"] 
    X_obs = context["X_obs"]
    
    # Use a simple fixed weight for exploitation vs exploration
    w_exploit = 0.75

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Estimate hypervolume improvement by resampling noisy predictions 
        n_samples = 100
        
        # Sample from the GP posterior distribution (assuming normality)
        samples = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"]  
            
            if std_val > 1e-8: # Avoid zero variance
                samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
            else:
                samples[:, i] = mean_val
                
        # Compute hypervolume for each sample (assuming all objectives are maximized)  
        
        hv_improvements = []
                
        ref_point_array = np.array([ref_point[name] for name in names])
                        
        for smp_idx, samp_obj_vals in enumerate(samples):
            if not any(np.isnan(samp_obj_vals)):
                # Compute hypervolume difference between reference and current sample
                # This is a simplified version that just looks at the dominated space  
                
                hv_improvement = 0.0
                
                front_points = context["pareto_front"]
                                
                ref_dominated_volume = np.prod(ref_point_array - samp_obj_vals)
                            
                if len(front_points) > 1:
                    # Simple approximation: compare to closest point on current pareto
                    min_dist_to_pf = float('inf')
                    
                    for pf_pt in front_points:
                        dist_sq = sum((samp_obj_vals[i] - pf_pt[i])**2 
                                      for i in range(len(names)))
                        
                        if dist_sq < min_dist_to_pf:  
                            min_dist_to_pf = dist_sq
                            
                    # Normalize by the scale of objectives to avoid bias towards larger scales    
                    
                hv_improvements.append(ref_dominated_volume)
                
        expected_hv_gain = np.mean(hv_improvements) if len(hv_improvements) > 0 else -1e6
        
        mu_sum_normed = sum(gp[name]["mean"] / front_range[name] for name in names)

        # Combine exploitation and HV gain estimate
        score = w_exploit * expected_hv_gain + (1.0-w_exploit)*mu_sum_normed

        scores.append(score)
    
    return scores