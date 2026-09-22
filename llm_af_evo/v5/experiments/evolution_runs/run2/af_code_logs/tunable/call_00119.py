def score_pool(context):
    """Score candidates based on how much they expand under-covered regions of the Pareto front by incorporating a coverage-gap term derived from nearest-front-point distances."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for meaningful density estimation
    pf = context['pareto_front']
    if len(pf) < 3:
        ref_points = context['Y_obs'] 
    else:  
        ref_points = pf
        
    scores = []
    k = 3
    
    # Compute squared distances from each candidate to reference points (front or obs)
    for cand in context["pool"]:
        x_cand = np.array(cand["x"])
        
        if len(ref_points) == 0:
            coverage_score = 0.0
        else:  
            dists_sq = []
            
            # For this candidate, compute squared Euclidean distances to each reference point 
            for pt in ref_points:
                obj_vals = np.array([pt[i] for i in range(len(names))]) if len(pt) == len(names) else None
                
                if obj_vals is not None:  # Ensure valid objective values
                    dist_sq = sum((x - y)**2 for x, y in zip(x_cand[:len(obj_vals)], obj_vals))
                    dists_sq.append(dist_sq)
                    
            # Sort distances and take mean of k nearest (or all available if less than k)  
            sorted_dists_sq = np.sort(dists_sq)[:k]
            
            coverage_score = 0.0
            if len(sorted_dists_sq) > 0:
                avg_dist_sq = sum(sorted_dists_sq)/len(sorted_dists_sq)
                
                # Normalize by range of objectives to make the score comparable across domains  
                ranges = [context["pareto_front_range"][name] for name in names]
            
                norm_avg_dist_sq = avg_dist_sq / (sum(ranges)**2) if sum(ranges)>0 else 1.0
                
                coverage_score = np.sqrt(norm_avg_dist_sq)
                
        # Blend with the acquisition value using a small weight  
        blended_score = context["pool"][i]["acq_value_norm"] + 0.1 * coverage_score
        
        scores.append(blended_score)

    return scores