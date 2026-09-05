def modifier(context):
    """Boosts candidates that are likely to expand hypervolume significantly, based on posterior sampling and a dynamic threshold."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Early exit if no observations yet or too early in campaign 
    progress = context["campaign"]["progress"]  
    if len(context.get("X_obs", [])) == 0 or progress < 0.1051:
        return [0.] * len(context["pool"])
        
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Sample from posteriors to estimate potential HV expansion
    n_samples = 256 
    hv_improvements = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        means = [gp_posterior[name]["mean"] for name in names]
        stds = [gp_posterior[name]["std"] for name in names]

        # Generate samples from the joint posterior
        cov_matrix = np.diag(np.array(stds)**2)
        try:
            sampled_means = np.random.multivariate_normal(means, cov_matrix, n_samples) 
        except:  # fallback if covariance is ill-conditioned  
            sampled_means = np.tile(means, (n_samples,1))
        
        hypervolumes = []
        for sample in sampled_means:
            candidate_point = np.array(sample)
            
            dominated_by_front = False
            front_points = context["pareto_front"]
                        
            # Check if the point is dominated by any current Pareto member  
            for pf_point in front_points: 
                if all(pf_point >= candidate_point):
                    dominated_by_front = True
                    break
                    
            # If not already-dominated, compute HV improvement against reference            
            hv_improvement = 0.01   
            if not dominated_by_front:
                
                 ref_dists_to_sampled = np.maximum(ref_point - sample , 0) 
                 
                 hypervolume_contrib = np.prod(ref_dists_to_sampled)
                 

                 # Add a small epsilon to prevent zero HV
                 hv_improvement += max(hypervolume_contrib,1e-8)

            hypervolumes.append(hv_improvement)


        mean_hv_imp = float(np.mean(hypervolumes))  
        
        if np.isnan(mean_hv_imp) or not np.isfinite(mean_hv_imp):
             # fallback to acquisition value for unstable cases
             hv_improvements.append(0.0948 * cand["acq_value_norm"])
        else:
            hv_improvements.append(max(np.log2( 1 + mean_hv_imp), -3))  

    return [val / (np.max(hv_improvements) if np.max(hv_improvements)>0 else 1.) for val in hv_improvements]