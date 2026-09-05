def score_pool(context):
    """Estimate Pareto front uncertainty via bootstrap resampling, then compute candidate hypervolume improvement over each resample."""
    import numpy as np
    
    # Bootstrap parameters
    n_resamples = 15
    names = context["objective_names"]
    
    # Reference point for HV calculation (already in correct orientation)
    ref_point = context['ref_point']
    
    scores = []
    Y_obs = context['Y_obs'] 
    
    def dominates(p, q):
        """Check if p weakly dominates q strictly in at least one objective."""
        return np.all(q >= p) and np.any(q > p)

    for cand in context["pool"]:
        # Candidate's predicted objectives
        pred_obj = [cand['gp_posterior'][name]['mean'] for name in names]
        
        total_improvement = 0.0
        
        # Resample Y_obs with replacement to estimate front uncertainty 
        for _ in range(n_resamples):
            resampled_Y = Y_obs[np.random.choice(Y_obs.shape[0], size=Y_obs.shape[0])]
            
            # Compute non-dominated set of the resample
            nondom_mask = np.ones(resampled_Y.shape[0], dtype=bool)
            for i, point_i in enumerate(resampled_Y):
                if not nondom_mask[i]:
                    continue  # Already marked as dominated 
                
                for j, point_j in enumerate(resampled_Y):  
                    if (i != j and nondom_mask[j] and dominates(point_i, point_j)):
                        nondom_mask[j] = False
            
            resample_nondom = resampled_Y[nodom_mask]
            
            # Compute HV of the original front
            hv_original = hypervolume(resample_nondom, ref_point)
                
            # Add candidate to this non-dominated set and compute new HV  
            extended_front = np.vstack([resample_nondom, pred_obj])
            hv_with_candidate = hypervolume(extended_front, ref_point) 
            
            total_improvement += (hv_with_candidate - hv_original)

        scores.append(total_improvement / n_resamples)
    
    return scores

def hypervolume(front, reference):
    """Compute the dominated hyper-volume of a front relative to a reference point."""
    if len(front) == 0:
        return 0.0
    
    # Sort by objectives in descending order (for efficient computation with WFG HV calc logic).
    sorted_front = np.array(sorted(front.tolist(), key=lambda x: [-xi for xi in x]))
    
    vol = 0
    prev_point = reference.copy()
    
    for point in reversed(sorted_front):
        contribution = 1.0 
        # Compute the volume of this box (from previous to current)
        for i, coord in enumerate(point):  
            if not np.isinf(coord) and not np.isnan(coord):
                contribution *= max(0., prev_point[i] - coord)

        vol += contribution
        # Update reference point along each dimension 
        for j, val in enumerate(prev_point):
            if (not np.isinf(val)) or (j < len(point)):
                new_val = min(val, point[j])
                assert not(np.isnan(new_val)), "NaN volume computation"
                prev_point[j] = new_val

    return vol