def score_pool(context):
    """Blend acquisition value with a coverage-gap score based on candidate's distance to nearest Pareto front points."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for k=3 neighbors
    pf = context.get("pareto_front", None)
    if pf is not None and len(pf) >= 3:
        ref_points = pf.copy()
    else:
        ref_points = context["Y_obs"].copy()

    scores = []
    
    # Precompute squared distances to avoid repeated sqrt operations (faster for large datasets)
    def compute_distance_sq(p1, p2):
        return np.sum((p1 - p2) ** 2)

    acq_values = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # For each candidate
    for i_cand, cand in enumerate(context["pool"]):
        
        gp_posterior = cand["gp_posterior"]
        
        pred_obj_vector = np.array([gp_posterior[name]["mean"] for name in names])
        
        if len(ref_points) == 0:
            coverage_gap_score = 0.0
        else: 
            # Compute distances to all reference points and get the k=3 smallest (or fewer)
            dists_sq = [compute_distance_sq(pred_obj_vector, ref_point) for ref_point in ref_points]
            
            sorted_dists_sq = np.sort(dists_sq)[:min(3, len(ref_points))]
    
            if len(sorted_dists_sq) == 0:
                coverage_gap_score = 0.0
            else: 
                 # Mean of k nearest distances (in original space)
                mean_dist = float(np.sqrt(np.mean(sorted_dists_sq)))
                
                # Normalize by front range to make score dimensionless and comparable across domains  
                frange = np.array([context["pareto_front_range"][name] for name in names])
    
                if not all(frange > 0):
                    coverage_gap_score = mean_dist
                else:
                     normalized_mean_dist = float(mean_dist / np.mean(frange))
                     
                     # Scale down to make it a secondary term (e.g., multiply by ~1/25)
                     coverage_gap_score = min(0.04 * normalized_mean_dist, 1.0) 
        
        total_score = acq_values[i_cand] + coverage_gap_score
        
        scores.append(total_score)

    return scores