def score_pool(context):
    """Estimates hypervolume improvement by resampling Pareto fronts and measuring candidate contribution to each."""
    import numpy as np
    
    # Set up bootstrap parameters  
    n_draws = 20 
    names = context["objective_names"]
    
    if len(names) == 1:
        return [cand['acq_value_norm'] for cand in context["pool"]]
        
    ref_point = np.array(context["ref_point"])
    y_obs = context["Y_obs"] 
    
    # Vectorized dominance test
    def is_dominated(points, p_idx):
        dominated_mask = (points[None] >= points[p_idx][None]) & \
                         (points[None] > points[p_idx][None])
        return np.any(np.all(dominanted_mask, axis=1))
        
    def get_pareto_front(y_vals):  # vectorized non-dominated sorting
        n_points = len(y_vals)
        is_nondom = np.ones(n_points, dtype=bool) 
        for i in range(n_points):
            if not is_nondom[i]:
                continue  
            dominated_mask = (y_vals[None] >= y_vals[i][None]) & \
                             (y_vals[None] > y_vals[i][None])
            dom_indices = np.where(np.all(dominated_mask, axis=1))[0]
            is_nondom[dom_indices] = False
        return y_vals[is_nondom]

    # For each candidate compute average hypervolume improvement over bootstrap draws 
    scores = []
    
    for cand in context["pool"]:
        
        mu = np.array([cand['gp_posterior'][name]['mean'] for name in names])
        hv_improvements = [] 
        
        for _ in range(n_draws):
            # Draw with replacement
            draw_indices = np.random.choice(len(y_obs), size=len(y_obs))
            y_bootstrapped = y_obs[draw_indices]
            
            front_with_cand  = get_pareto_front(np.vstack([y_bootstrapped, mu]))
            front_without_cand = get_pareto_front(y_bootstrapped)
            
            hv_before_adding_candidate = np.prod(ref_point - 
                                                 np.max(front_without_cand,axis=0))
            hv_after_adding_candidate =  np.prod(ref_point -
                                                np.max(front_with_cand ,axis=0)) 
            
            if len(front_with_cand) > len(front_without_cand):
                improvement = max(0.,hv_after_adding_candidate - 
                                  hv_before_adding_candidate)
                
            else:
                # If no new point was added, the HV should not change.
                improvement = 0. 

            
            hv_improvements.append(improvement)

        avg_hv_imp = np.mean(hv_improvements) 
        
        scores.append(avg_hv_imp * cand['acq_value_norm'])
        
    return scores