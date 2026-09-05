def modifier(context):
    """Estimate hypervolume expansion potential based on posterior sampling and penalize near-duplicates."""
    import numpy as np
    
    pool = context["pool"]
    Y_obs = context["Y_obs"] 
    pareto_front = context["pareto_front"]
    ref_point = context["ref_point"]
    
    n_samples = 50
    names = context['objective_names']
    front_range = context["pareto_front_range"]

    # Compute reference point normalized by front range for consistent scaling  
    ref_norm = np.array([context["ref_point_by_name"][name] / front_range[name] 
                         for name in names])
    
    values = []
    
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Sample from the joint GP distribution (assuming independent objectives)
        samples = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"] 
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)

        # Compute hypervolume improvement estimate from sampled points
        hv_improvements = []
        
        for sample in samples:
            if all(sample >= ref_norm):  # Only consider feasible candidates  
                continue
                
            # Estimate HV contribution of this point (simplified)
            diff_to_ref = np.maximum(ref_norm - sample, 0.0) 
            hv_contrib = np.prod(diff_to_ref)

            # Check dominance and adjust for existing front
            dominates_any = False
            
            if len(pareto_front) > 0:
                for pf_point in pareto_front:  
                    if all(sample <= pf_point):   # dominated by PF point, no HV gain 
                        hv_contrib = 0.0                        
                        break
                        
                    elif np.all(pf_point < sample):
                        dominates_any = True
                  
            if not dominates_any and len(pareto_front) > 0:
                for i in range(len(ref_norm)):
                   diff_to_pf_i = max(0, pf_point[i] - sample[i])  
                
           hv_improvements.append(hv_contrib)
            
        expected_hv_imp = np.mean(hv_improvements)

        # Near-duplicate suppression based on observed points
        cand_x = cand["x"]
        
        min_dist_to_observed = float('inf')
 
        for obs in Y_obs:
            dist = np.linalg.norm(obs - cand['gp_posterior'])
            
            if dist < min_dist_to_observed:  
                min_dist_to_observed = dist
                
       # Penalize near-duplicates (this suppresses redundant proposals)
        
        penalty_factor = 0.1 * max(0, 1 - min_dist_to_observed) 
        
        values.append(expected_hv_imp + penalty_factor)

    return values