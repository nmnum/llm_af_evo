def score_pool(context):
    """Estimate hypervolume expansion potential by sampling noisy predictions and scoring based on how much they improve over current front."""
    import numpy as np
    
    names = context["objective_names"]
    pf = context["pareto_front"] 
    ref_point = context["ref_point"]

    scores = []
    
    # For each candidate, sample from its posterior to estimate hypervolume improvement
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample multiple noisy predictions (10 samples per objective)
        n_samples = 10 
        sampled_objs = []  
        for _ in range(n_samples):
            sample_obj = []
            for name in names:
                mean, std = gp_posterior[name]["mean"], gp_posterior[name]["std"]
                # Sample from normal distribution
                s = np.random.normal(mean, std)
                sample_obj.append(s) 
            sampled_objs.append(sample_obj)

        # Compute hypervolume improvement of each sample against current front  
        hv_improvements = []
        
        for obj in sampled_objs:
            # Create candidate point with noise added to its mean
            cand_point = np.array(obj)
            
            # Check if this noisy version would dominate the reference point (i.e. is better than all objectives) 
            dominates_ref = True  
            for i, val in enumerate(cand_point):
                if val <= ref_point[i]:
                    dominates_ref = False
                    break
                    
            hv_improvement = 0.
            
            # If it's actually improving on the reference point (better than every objective)
            if dominates_ref:
                
                # Compute hypervolume of union with current front, then subtract original HV 
                expanded_front = np.vstack([pf, cand_point])
                    
                # Simple approximation: assume all objectives are maximized and compute volume
                try:
                    vol_expanded = 1.0  
                    for i in range(len(names)):
                        max_val_infront = np.max(expanded_front[:,i]) if len(expanded_front) > 1 else ref_point[i]
                        
                        # If our candidate point is better than the reference, we add to volume
                        diff_to_ref = (max_val_infront - cand_point[i])
                        vol_expanded *= max(diff_to_ref + np.finfo(float).eps,0)
                    hv_improvement += 1. * vol_expanded 
                except:
                    pass
                    
            # Average across samples  
            hv_improvements.append(hv_improvement)

        estimated_hvi = sum(hv_improvements) / len(hv_improvements)
        
        scores.append(estimated_hvi + cand["acq_value_norm"])
    
    return scores