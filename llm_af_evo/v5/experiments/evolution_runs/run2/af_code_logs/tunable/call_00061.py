def score_pool(context):
    """Resample candidate objectives under Gaussian noise and compute expected hypervolume gain for robust exploration-exploitation."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Use a fixed number of noisy samples to estimate HV improvement
    n_samples = 50
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample from the joint posterior (approximated by drawing per-objective)
        sampled_objs = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            
            if not isinstance(mean_val, float) or not isinstance(std_val, float):
                # Fallback to deterministic
                sampled_objs[:,i] = mean_val  
            else:
                samples_i = np.random.normal(loc=mean_val, scale=max(1e-8, std_val), size=n_samples)
                sampled_objs[:, i] = samples_i
                
        # For each sample compute the HV contribution relative to current front 
        hv_contributions = []
        
        for s_idx in range(n_samples):
            obj_s = sampled_objs[s_idx]
            
            if np.all(obj_s <= ref_point):  # dominated by reference
                contrib = -np.sum((ref_point - obj_s) ** 2)
                
            else:
                front_points = context["pareto_front"]
                dominates_any = False
                
                for fpt in front_points: 
                    if all(fpt[i] >= obj_s[i] and not np.isclose(obj_s[i],fpt[i]) \
                           for i in range(len(names))):
                        # This sample point is dominated by some existing Pareto
                        dominates_any = True  
                        break
                        
                contrib = 0.1 * (np.sum((ref_point - obj_s) ** 2)) if not dominates_any else -5e-3
                
            hv_contributions.append(contrib)
            
        expected_hv_improvement = np.mean(hv_contributions)

        # Blend with acquisition value
        acq_val_normed = cand["acq_value_norm"]
        
        final_score = (0.7 * acq_val_normed + 
                       0.3 * max(expected_hv_improvement, -1e-6)) 

        scores.append(final_score)
    
    return scores