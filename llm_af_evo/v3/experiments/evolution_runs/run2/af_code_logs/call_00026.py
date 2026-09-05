def score_pool(context):
    """Estimate hypervolume expansion potential using Monte Carlo-sampled GP posteriors for robust acquisition."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    # Sample from each candidate's posterior to estimate HV improvement
    n_samples = 50
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        samples = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"]
            
            # Draw noisy sample from GP posterior
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
            
        # Calculate hypervolume improvement estimate over reference point  
        hv_improvements = []
        
        for s in samples:
            if all(s[i] <= ref_point[i] for i in range(len(names))):
                # Point is dominated by current front
                continue
                
            # Compute HV contribution of this sample 
            contrib_hv = 1.0
            
            try:  
                for j, val in enumerate(s):
                    diff = max(0., ref_point[j] - val)
                    if not np.isclose(diff, 0.) and not np.isnan(diff):    
                        contrib_hv *= diff
                            
                hv_improvements.append(contrib_hv) 
            except:
                continue
                
        # Use mean HV improvement as score (higher is better)
        avg_hv = np.mean(hv_improvements) if len(hv_improvements) > 0 else -1e6
        
        scores.append(avg_hv)

    return scores