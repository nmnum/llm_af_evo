def score_pool(context):
    """Estimate hypervolume improvement using bootstrap resampling of observations to compute non-dominated fronts, then average candidate contribution across these front estimates."""
    import numpy as np
    
    # Parameters for bootstrapping
    n_bootstrap = 20
    ref_point = context["ref_point"]
    
    def dominates(p, q):
        return np.all(q >= p) and np.any(q > p)
        
    def is_non_dominated(points):
        n_points = len(points)
        nd_mask = np.ones(n_points, dtype=bool)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if dominates(points[i], points[j]):
                    nd_mask[j] = False
                elif dominates(points[j], points[i]):
                    nd_mask[i] = False
        return nd_mask
    
    def hypervolume(front, ref_point):
        # Simple implementation for two objectives; extendable to higher dims via recursive formula or libraries like pymoo if needed.
        front_sorted_by_f1_descending = sorted(front, key=lambda x: -x[0])
        
        vol_sum = 0.0
        prev_x = float('inf')
        y_min = min(p[1] for p in front) 
        # For each segment of the pareto curve:
        for i, (f1_val, f2_val) in enumerate(front_sorted_by_f1_descending):
            if not np.isfinite(f1_val): continue
            vol_sum += max(0.0, ref_point[0] - prev_x) * abs(ref_point[1] - y_min)
            # Update the minimum Y value seen so far from this point onwards.
            for j in range(i+1, len(front_sorted_by_f1_descending)):
                if front_sorted_by_f1_descending[j][0] <= f1_val:
                    break
                else: 
                    new_y = min(y_min,front_sorted_by_f1_descending[j][1])
                    
        return vol_sum
    
    # Bootstrap sampling and compute average hypervolume improvement per candidate.
    
    scores = []
    for cand in context["pool"]:
        
        gp_posterior = cand['gp_posterior']
        pred_mean = np.array([gp_posterior[name]["mean"] for name in ["f1", "f2"]])
                
        total_improvement = 0.0
        n_valid_draws = 0
        
        # Draw bootstrap samples from Y_obs and compute non-dominated fronts.
        X_obs, Y_obs = context["X_obs"], context["Y_obs"]
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(Y_obs), size=len(Y_obs))
            
            y_sampled = Y_obs[indices]
                
            # Compute the set of points that are non-dominated by any other point
            nd_mask = is_non_dominated(y_sampled)
            front_samples = y_sampled[nd_mask] 
            
            if len(front_samples) == 0:
                continue
                
            n_valid_draws +=1
            
            
            hv_without_cand = hypervolume(front_samples, ref_point)

            # Add candidate to the sampled non-dominated set
            extended_front = np.vstack([front_samples, pred_mean])
                
            nd_mask_extended = is_non_dominated(extended_front)
            front_with_candidate = extended_front[nd_mask_extended]
            
            hv_with_cand = hypervolume(front_with_candidate, ref_point)

            total_improvement += (hv_with_cand - hv_without_cand) 
            
        if n_valid_draws > 0:
            avg_hv_imp = float(total_improvement / n_valid_draws)
        else: 
            # fallback in case all draws resulted in empty fronts
            avg_hv_imp = 0.0
            
        scores.append(avg_hv_imp)

    return scores