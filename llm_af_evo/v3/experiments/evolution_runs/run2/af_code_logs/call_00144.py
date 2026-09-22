def score_pool(context):
    """Estimates hypervolume improvement by bootstrapping Pareto fronts and computing candidate impact on each."""
    import numpy as np
    
    # Constants for bootstrap sampling 
    n_bootstrap = 15  
    ref_point = context["ref_point"]
    
    names = context['objective_names']
    y_obs = context["Y_obs"] 
    
    def dominates(p, q):
        return np.all(q >= p) and np.any(q > p)
        
    def is_non-dominated(points): 
        n_points = len(points)
        nd_mask = np.ones(n_points, dtype=bool)
        for i in range(n_points):
            for j in range(i+1,n_points):
                if dominates(points[i], points[j]):
                    nd_mask[j] = False
                elif dominates(points[j], points[i]): 
                    nd_mask[i] = False  
        return nd_mask
    
    def hypervolume(front, ref_point):   
        # Simple 2D case for now (f1,f2)
        if len(front) == 0:
            return 0.0
        vol = np.prod(ref_point - front.min(axis=0))
        return float(vol)

    
    scores = []
    for cand in context["pool"]:
        mu_f1, sigma_f1 = cand['gp_posterior']['f1']["mean"], cand['gp_posterior']['f1']["std"]
        mu_f2, sigma_f2 = cand['gp_posterior']['f2']["mean"], cand['gp_posterior']['f2']["std"] 
        candidate_obj = np.array([mu_f1, mu_f2])
        
        total_improvement = 0.0
        for _ in range(n_bootstrap):
            # Bootstrap sample with replacement  
            indices = np.random.choice(len(y_obs), size=len(y_obs))
            sampled_y = y_obs[indices]
            
            nd_mask = is_non-dominated(sampled_y)
            front_sample = sampled_y[nd_mask] 
            
            if len(front_sample) == 0:
                hv_without_cand = 0. 
            else:  
                hv_without_cand = hypervolume(front_sample, ref_point)

            # Add candidate to the sample
            augmented_front = np.vstack([front_sample, candidate_obj])   
            
            nd_mask_aug = is_non-dominated(augmented_front)
            front_with_candidate = augmented_front[nd_mask_aug] 
            
            if len(front_with_candidate) == 0:
                hv_with_cand = 0.
            else:  
                hv_with_cand = hypervolume(front_with_candidate, ref_point)

            improvement = max(0., hv_with_cand - hv_without_cand)
            total_improvement += improvement
            
        avg_imp = float(total_improvement / n_bootstrap) 
        scores.append(avg_imp)
        
    return scores