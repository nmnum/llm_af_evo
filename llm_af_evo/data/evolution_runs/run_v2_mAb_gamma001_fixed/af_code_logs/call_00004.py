def score_pool(context):
    """Estimate hypervolume improvement using bootstrap resampling of observations to compute non-dominated fronts, then average candidate improvements across these front estimates."""
    import numpy as np
    
    n_resamples = 20
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
    
    def hypervolume(point, ref_point_):
        diff = np.maximum(ref_point_ - point, 0)
        return np.prod(diff)

    scores = []
    
    for cand in context["pool"]:
        gp_pred = cand["gp_posterior"]
        pred_obj = [gp_pred[name]["mean"] for name in context["objective_names"]]
        
        total_improvement = 0.0
        n_fronts_considered = 0
        
        # Bootstrap resampling of observations to estimate Pareto front uncertainty.
        Y_obs = context['Y_obs']
        for _ in range(n_resamples):
            indices = np.random.choice(len(Y_obs), size=len(Y_obs), replace=True)
            sampled_Y = Y_obs[indices]
            
            # Compute non-dominated points from this sample
            nd_mask = is_non_dominated(sampled_Y)
            front_sample = sampled_Y[nd_mask]

            if len(front_sample) == 0:
                continue
                
            n_fronts_considered += 1
            
            hv_before_adding_cand = hypervolume(np.min(front_sample, axis=0), ref_point)

            # Add candidate to the sample and recompute
            extended_front = np.vstack([front_sample, pred_obj])
            
            nd_mask_extended = is_non_dominated(extended_front)
            front_with_candidate = extended_front[nd_mask_extended]
                
            hv_after_adding_cand = hypervolume(np.min(front_with_candidate, axis=0), ref_point)

            total_improvement += (hv_after_adding_cand - hv_before_adding_cand) / n_resamples
            
        scores.append(total_improvement)
        
    return scores