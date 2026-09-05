def modifier(context):
    """Resample posterior means with Gaussian noise and compute expected domination gain over current front."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    n_obj = len(names)
    values = []
    
    # Use a fixed seed for reproducibility of resampling within same campaign
    np.random.seed(42) 
    
    # Sample from each candidate's GP posterior 10 times to estimate mean domination gain 
    nsamples = 10
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = []
        for _ in range(nsamples):
            sample_point = [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) 
                            for name in names]
            samples.append(sample_point)
            
        # Compute how many of these noisy samples would dominate the current front
        n_dominated = 0
        
        for samp in samples:
            is_dom_by_front = False
            
            for pf_point in context["pareto_front"]:
                dominates_pf = True
                
                for i, obj_val in enumerate(samp):
                    if not (obj_val >= pf_point[i]):
                        dominates_pf = False
                        break
                        
                # If this sample point dominates the current front point 
                if dominates_pf:
                    is_dom_by_front = True
                    n_dominated += 1
                    
        expected_domination_gain = float(n_dominated) / nsamples
        
        values.append(expected_domination_gain)
        
    return values