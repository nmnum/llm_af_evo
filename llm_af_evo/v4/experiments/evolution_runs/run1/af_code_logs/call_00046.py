def score_pool(context):
    """Estimate uncertainty in Pareto front location via bootstrap resampling; for each candidate, compute average hypervolume improvement across resamples."""
    import numpy as np
    
    # Setup
    n_resample = 15
    ref_point = context["ref_point"]
    Y_obs = context["Y_obs"] 
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
        # Simple 2D case for clarity; generalization to higher dims is more complex but doable.
        if len(front) == 0:
            return 0.0
            
        front_sorted_f1 = sorted(front[:, 0], reverse=True)
        front_sorted_f2 = [front[i, 1] for i in range(len(front)) 
                           if np.any([dominates(np.array([[f1,f2]]), front[j])  
                                      for j, (f1,f2) in enumerate(zip(front_sorted_f1[:i]+front_sorted_f1[i+1:],   
                                                                      [front[k, 1] for k in range(len(front)) 
                                                                       if np.any([dominates(np.array([[front_sorted_f1[i], front[j][0]]),  
                                                                                                   point)]) and j < i]))])]

        # Use the exact definition: hypervolume = product of distances to reference
        vol_total = 1.0
        for obj_idx in range(len(ref_point)):
            if len(front) > 0:
                max_val = np.max([p[obj_idx] for p in front])
                diff_to_ref = ref_point[obj_idx] - max_val  
                assert(diff_to_ref >= 0), "Reference point must dominate all points."
                vol_total *= diff_to_ref
        return vol_total

    # Resample and compute non-dominated sets 
    hv_improvements = []
    
    for _ in range(n_resample):
        
        indices = np.random.choice(len(Y_obs), size=len(Y_obs))
        resampled_Y = Y_obs[indices]
                
        nd_mask = is_non_dominated(resampled_Y)
        front_nd = resampled_Y[nd_mask]

        # Compute HV improvement of each candidate
        cand_hv_improvements = []
        
        for cand in pool:
            pred_f1, pred_f2 = cand["gp_posterior"]["f1"]["mean"], cand["gp_posterior"]["f2"]["mean"]
            
            hv_without_cand = hypervolume(front_nd, ref_point)
                        
            # Add candidate to front and recompute HV
            new_front = np.vstack([front_nd, [pred_f1, pred_f2]])
                
            nd_mask_new = is_non_dominated(new_front) 
            front_with_cand = new_front[nd_mask_new]
            
            hv_with_cand = hypervolume(front_with_cand, ref_point)
                        
            # Improvement: HV with cand minus without
            improvement = max(0.0, hv_with_cand - hv_without_cand)

            cand_hv_improvements.append(improvement) 
            
        hv_improvements.append(cand_hv_improvements)


    avg_scores = np.mean(hv_improvements, axis=0)
    
    return list(avg_scores)