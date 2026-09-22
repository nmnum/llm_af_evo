def score_pool(context):
    """Estimate Pareto front novelty by bootstrapping observed points and measuring candidate hypervolume improvement across resampled fronts."""
    import numpy as np
    
    # Parameters
    n_bootstrap = 15
    ref_point = context["ref_point"]
    
    # Helper: vectorized dominance test for maximization
    def is_dominated(points, p_idx):
        p = points[p_idx]
        dominated_mask = np.all(points >= p, axis=1) & np.any(points > p, axis=1)
        return dominated_mask
    
    def get_non-dominated_front(points):
        n_points = len(points)
        if n_points == 0:
            return []
        
        # For each point check dominance
        is_dom = [is_dominated(points, i) for i in range(n_points)]
        not_dominated = ~np.any(is_dom, axis=1)
        return np.where(not_domiated)[0]
    
    def hypervolume(point_set, reference):
        if len(point_set) == 0:
            return 0.0
        # Simple implementation: assumes all objectives are maximized and ref is lower bound.
        # We compute the product of differences between each point's objective values 
        # and corresponding components in `reference`, for non-dominated points only,
        # but this simple version works under assumption that we're dealing with a single reference
        if len(point_set) == 1:
            return np.prod(np.maximum(reference - point_set[0], 0))
        
        volume = 0.0 
        for p in point_set:  
             diff = (reference-p)
             vol_contrib = np.product(diff[np.greater_equal(diff, 0)])
             if not np.isnan(vol_contrib) and vol_contrib > 1e-12:
                 volume += vol_contrib
                 
        return float(volume)

    # Bootstrap resampling 
    Y_obs = context["Y_obs"]
    
    scores = []
    for cand in context["pool"]:
        
        improvement_sum = 0.0
        
        n_cand_points = len(Y_obs)
            
        for _ in range(n_bootstrap):
            sampled_indices = np.random.choice(np.arange(len(Y_obs)), size=n_cand_points, replace=True)  
            bootstrap_front_Y = Y_obs[sampled_indices]
                
            # Compute non-dominated front from this resample
            nd_idxes = get_nonominated_front(bootstrap_front_Y)
            
            if len(nd_idxes)==0:
                continue
                
            frontier_without_candidate= bootstrap_front_Y[nd_idxes] 
                  
            # Add candidate to the set of points and recompute its contribution  
            cand_point_y  = np.array([cand["gp_posterior"][name]["mean"] for name in context['objective_names']])
            
            front_with_cand   = np.vstack((frontier_without_candidate, cand_point_y))
              
            vol_before_adding= hypervolume(frontier_without_candidate, ref_point) 
            vol_after_adding  =hypervolume( front_with_cand ,ref_point)
                
            improvement_sum += (vol_after_adding - vol_before_adding)

        avg_improvement = improvement_sum / n_bootstrap if n_bootstrap >0 else float('nan')
        
        scores.append(avg_improvement)  
    return np.array(scores).tolist()