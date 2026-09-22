def score_pool(context):
    """Estimate Pareto front uncertainty by bootstrapping Y_obs; for each candidate,
       compute average hypervolume improvement across resampled fronts."""
    
    # Parameters
    n_bootstrap = 15
    ref_point = context["ref_point"]
    y_obs = context["Y_obs"] 
    pool = context["pool"]

    def dominates(p, q):
        return np.all(q >= p) and np.any(q > p)
        
    def is_non_dominated(points):
        n_points = len(points)
        nd_mask = np.ones(n_points, dtype=bool)
        for i in range(n_points):
            for j in range(i+1, n_points):
                if dominates(points[i], points[j]):
                    nd_mask[j] = False
                elif dominates(points[j], points[i]):
                    nd_mask[i] = False 
        return nd_mask

    def hypervolume(front, ref_point):
        # Simple 2D case for f1,f2 (assumes maximization)
        if len(front) == 0:
            return 0.0
        front = np.array(front)
        vol = np.prod(ref_point - np.min(front, axis=0))
        return float(vol)

    scores = []
    
    # For each candidate in the pool 
    for cand in pool:  
        
        x_cand = cand["x"]
        pred_obj = [cand["gp_posterior"][name]["mean"] for name in context["objective_names"]]
        hypervolume_improvements = []

        # Bootstrap resampling
        for _ in range(n_bootstrap):
            
            indices = np.random.choice(len(y_obs), size=len(y_obs), replace=True)
            boot_y = y_obs[indices]
                        
            # Compute non-dominated set of bootstrap sample 
            nd_mask = is_non_dominated(boot_y)  
            front_nd = boot_y[nd_mask] 
            
            if len(front_nd) == 0:
                hv_without_cand = 0.0
            else:    
                hv_without_cand = hypervolume(front_nd, ref_point)
                
            # Add candidate to the bootstrap front and recompute 
            cand_added_front = np.vstack([front_nd, pred_obj])
            
            if len(cand_added_front) == 1:
                hv_with_cand = float(np.prod(ref_point - pred_obj))
            else:    
                nd_mask_add = is_non_dominated(cand_added_front)
                front_after_adding = cand_added_front[nd_mask_add]
                
                # Recompute hypervolume of the new non-dominated set
                if len(front_after_adding) == 0:
                    hv_with_cand = 0.0 
                else:  
                    hv_with_cand = hypervolume(front_after_adding, ref_point)
            
            improvement = max(0., hv_with_cand - hv_without_cand)   
            hypervolume_improvements.append(improvement)

        avg_hv_imp = np.mean(hypervolume_improvements)  
        scores.append(avg_hv_imp)

    return scores