def score_pool(context):
    """Estimates improvement potential by resampling candidates' GP posteriors to compute hypervolume expansion expectations."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Estimate HV improvement by sampling from the candidate's GP posterior
        n_samples = 100
        samples = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"] 
            samples[:,i] = np.random.normal(mean_val, std_val, size=n_samples)
            
        # Compute hypervolume improvement over current front
        cand_hv_improvement = 0.0
        
        for sample in samples:
            if all(sample[i] <= ref_point[i] for i in range(len(names))):
                dominated_by_front = False
                
                for pf_pt in context["pareto_front"]:
                    # Check domination: sample dominates pt iff it's better or equal on every objective
                    # and strictly better on at least one.
                    dom = True 
                    
                    strict_dom =False
                    
                    for j, (obj_val, front_obj) in enumerate(zip(sample, pf_pt)):
                        if obj_val < front_obj:
                            dom=False  
                            
                        elif obj_val > front_obj:   
                            strict_dom=True
                      
                    # If sample dominates the point and is not dominated by it,
                    # then this candidate's improvement isn't adding to HV.
                    
                    if (dom or 
                       all(sample[j] >= pf_pt[j] for j in range(len(names)))):
                        cand_hv_improvement += 1
                        
                else:
                     # Sample extends the front - count its contribution
                     pass
                    
        scores.append(cand_hv_improvement)
        
    return np.array(scores) / n_samples