def score_pool(context):
    """Score candidates based on acquisition value blended with a coverage-gap term targeting sparse regions of the Pareto front."""
    import numpy as np
    
    names = context["objective_names"]
    pool = context["pool"]
    
    # Prepare for distance calculations
    if len(pool) == 0:
        return []
        
    pf = context["pareto_front"] 
    y_obs = context["Y_obs"]

    scores = []

    k_nearest = min(3, len(pf))
    if k_nearest < 3:  
        # Use all observations as proxy for front density
        ref_points = y_obs.copy()
    else:
        ref_points = pf

    acq_values = np.array([cand["acq_value_norm"] for cand in pool])

    for i, cand in enumerate(pool):
        
        pred_obj_vec = [cand["gp_posterior"][name]["mean"] for name in names]
 
        # Compute distances to reference points
        dists_to_ref = []
        if len(ref_points) > 0:
            ref_mat = np.array(ref_points)
            vec_tile = np.tile(pred_obj_vec, (len(ref_points),1))
            
            diff_sq = (vec_tile - ref_mat)**2 
            euclid_dists = np.sqrt(np.sum(diff_sq,axis=1)) 
            
            dists_to_ref.extend(euclid_dists.tolist())
        
        # Get k nearest distances
        if len(dists_to_ref) > 0:
           sorted_distances = sorted(dists_to_ref)
           mean_dist = sum(sorted_distances[:k_nearest]) / float(k_nearest)
        else:  
            mean_dist = 1.0   # fallback in case no reference points exist

        
        coverage_gap_score = (mean_dist ** -2) if mean_dist > 0 else 1e6  

       
        blended_acq_value_with_coverage = acq_values[i] + 0.3 * coverage_gap_score
        

    
        scores.append(blended_acq_value_with_coverage)

    return scores