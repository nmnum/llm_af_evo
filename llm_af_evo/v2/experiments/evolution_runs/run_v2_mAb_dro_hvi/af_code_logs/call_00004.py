def score_pool(context):
    """Estimate hypervolume improvement by bootstrapping Pareto fronts and measuring candidate impact on dominated volume."""
    import numpy as np
    
    # Bootstrap parameters
    n_bootstrap = 20
    objective_names = context["objective_names"]
    
    def dominates(p, q):
        return np.all(q >= p) and np.any(q > p)
        
    def is_non_dominated(points):
        n_points = points.shape[0]
        non_dom = np.ones(n_points, dtype=bool)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if dominates(points[i], points[j]):
                    non_dom[j] = False
                elif dominates(points[j], points[i]):
                    non_dom[i] = False
        return non_dom
    
    # Prepare reference point and objectives
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        mu = np.array([cand["gp_posterior"][name]["mean"] 
                       for name in objective_names])
        
        total_improvement = 0.0
        
        # Bootstrap sampling and hypervolume computation
        Y_obs = context["Y_obs"]
        n_samples = len(Y_obs)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True) 
            boot_Y = Y_obs[indices]
            
            non_dom_mask = is_non_dominated(boot_Y)
            front_points = boot_Y[non_dom_mask] 
            
            # Compute hypervolume of current front
            hv_front = 0.0  
            if len(front_points):
                diffs = np.maximum(ref_point - front_points, 0) 
                hv_front = np.prod(diffs.sum(axis=1))
            
            # Add candidate to the bootstrapped front and compute new HV
            extended_front = np.vstack([front_points, mu])
            non_dom_mask_extended = is_non_dominated(extended_front)
            ext_front_pts = extended_front[non_dom_mask_extended]
                
            hv_ext = 0.0  
            if len(ext_front_pts):
                diffs_ext = np.maximum(ref_point - ext_front_pts, 0) 
                hv_ext = np.prod(diffs_ext.sum(axis=1))
            
            # Improvement is the difference in hypervolume
            improvement = max(0., hv_ext - hv_front)
            total_improvement += improvement
            
        avg_imp = total_improvement / n_bootstrap  
        scores.append(avg_imp)

    return scores