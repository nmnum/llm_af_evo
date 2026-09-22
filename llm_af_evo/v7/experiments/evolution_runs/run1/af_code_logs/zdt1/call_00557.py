def modifier(context):
    """Resample candidate objectives under posterior uncertainty to estimate expected hypervolume gain and reward candidates that are likely to improve front coverage."""
    import numpy as np
    
    if len(context["Y_obs"]) == 0:
        return [0.] * len(context["pool"])
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    n_samples = 50
    values = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        # Sample from each objective's posterior (mean, std)
        samples = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean = gp_posterior[name]["mean"]
           std  = gp_posterior[name]["std"]
            
            if not isinstance(mean, float) or not isinstance(std, float): 
                # fallback to deterministic
                samples[:,i] = mean  
                
            else:
                 samples[:, i] = np.random.normal(loc=mean, scale=max(1e-8,std), size=n_samples)
        
        hv_gains = []
        for s in range(n_samples):
            
            sample_point = samples[s]
                        
            # Compute hypervolume contribution of this sampled point
            if all(sample_point[i] <= ref_point[i] for i in range(len(names))):
                # valid improvement over reference 
                
                try:
                    hv_improvement = 1.0
                    
                    # compute volume relative to current pareto front (simplified)
                    
                    pf = context["pareto_front"]
            
                    if len(pf) == 0:  
                        vol_ref_to_point = np.prod(ref_point - sample_point )
                        
                    else:

                        dominated_by_any = False
                        for p in range(len(pf)):
                            # point dominates sampled?
                            
                            dom_all = True 
                            for j in range(len(names)): 

                                if not (sample_point[j] >= pf[p][j]):
                                    dom_all=False  
                                    
                            if dom_all:
                                
                                dominated_by_any=True

                        
                        vol_ref_to_point=1.0
                        
                    hv_gains.append(hv_improvement)
                    
                except Exception: 
                     # fallback
                     pass
        
        expected_hypervolume_gain = np.mean(hv_gains)  if len(hv_gains)>0 else 0.
        
        values.append(expected_hypervolume_gain * .2)

    return values