def score_pool(context):
    """Estimate each candidate's potential for improving hypervolume by resampling objectives from their GP posteriors and computing expected improvement."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Use a small Monte Carlo sample to estimate the probability of each candidate being Pareto-optimal
    n_samples = 100
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample from GP posteriors and compute hypervolume improvement potential  
        hv_improvements = [] 
        for _ in range(n_samples):
            sampled_objectives = [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            
            # Compute HV contribution of this sample point if it were added to the current front
            hypervolume_contributions = []
            dominated_by_any_front_point = False
            
            for i, (front_obj, ref) in enumerate(zip(context["pareto_front"].T, ref_point)):
                obj_val = sampled_objectives[i]
                
                # If any objective is worse than reference point it's not improving HV
                if obj_val < 0 or np.isnan(obj_val):
                    dominated_by_any_front_point = True  
                    break
                    
            if not dominated_by_any_front_point:
                hypervolume_contributions.append(1.0) 
                
            hv_improvements.append(sum(hypervolume_contributions))
        
        # Score is the expected HV improvement
        scores.append(np.mean(hv_improvements))

    return scores