def score_pool(context):
    """Estimates hypervolume improvement by resampling Pareto fronts and computing candidate's contribution to each."""
    import numpy as np

    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    # Bootstrap sampling of Y_obs (with replacement)
    n_bootstrap = min(20, max(10, len(context['Y_obs']) // 4))
    scores = []
    for cand in context["pool"]:
        improvement_sum = 0.0
        num_samples = 0
        
        gp_posterior = cand["gp_posterior"]
        
        # Generate candidate's predicted objectives (mean)
        y_candidate = np.array([gp_posterior[name]["mean"] for name in names])
        
        # Resample Y_obs with replacement and compute hypervolume improvement 
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(context['Y_obs']), size=len(context['Y_obs']))
            sample_Y = context["Y_obs"][indices]
            
            if len(sample_Y) == 0:
                continue
                
            # Vectorized non-dominated sorting
            dominated_mask = np.zeros(len(sample_Y), dtype=bool)
            for i in range(len(sample_Y)):
                dominates_i = (sample_Y >= sample_Y[i]).all(axis=1) & \
                              (sample_Y > sample_Y[i]).any(axis=1)
                if dominates_i.any():
                    dominated_mask[i] = True
            
            # Non-dominated points
            front_sample = sample_Y[~dominated_mask]
            
            if len(front_sample) == 0:
                continue
                
            num_samples += 1

            hypervolume_without_candidate = _hypervolume_calculate(front_sample, ref_point)
                
            combined_front = np.vstack([front_sample, y_candidate])
                        
            # Recalculate non-dominated front including candidate
            dominated_mask_new = np.zeros(len(combined_front), dtype=bool) 
            for i in range(len(combined_front)):
                dominates_i = (combined_front >= combined_front[i]).all(axis=1) & \
                              (combined_front > combined_front[i]).any(axis=1)
                if dominates_i.any():
                    dominated_mask_new[i] = True
                    
            front_with_candidate = combined_front[~dominated_mask_new]
            
            hypervolume_with_candidate = _hypervolume_calculate(front_with_candidate, ref_point) 
            
            improvement_sum += (hypervolume_with_candidate - hypervolume_without_candidate)
        
        scores.append(improvement_sum / num_samples if num_samples > 0 else 0.0)

    return scores


def _hypervolume_calculate(points, reference):
    """Simple hyper-volume calculation assuming maximization and dominated points."""
    
    # If no or single point
    if len(points) == 0:
        return 0
    
    n_obj = points.shape[1]
        
    def dominates(p, q): 
        """
        p is weakly better than (dominates) q in all objectives,
        and strictly better on at least one.
        """
        # This returns True if every element of 'p' >= corresponding element
        # of 'q', AND there's a strict improvement somewhere.  
        
        return np.all(p >= q) & any(p > q)
    
    hypervolume = 0
    
    for point in points:
        volume_contributed = 1 
        for i, ref_val in enumerate(reference):
            if point[i] < ref_val: # Only consider contributions when not dominated
                vol_diff = (ref_val - max(point[i], reference[i]))  
                volume_contributed *= vol_diff
                
        hypervolume += volume_contributed
    
    return hypervolume