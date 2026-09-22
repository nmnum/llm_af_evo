def score_pool(context):
    """Score candidates based on acquisition value blended with a coverage-gap term reflecting under-covered regions of the Pareto front."""
    import numpy as np
    
    pool = context["pool"]
    pareto_front = context["pareto_front"] 
    Y_obs = context["Y_obs"]
    
    # Determine which points to use for nearest neighbor search
    if len(pareto_front) >= 3:
        ref_points = pareto_front
    else:
        ref_points = Y_obs
    
    scores = []
    k = 3
        
    for cand in pool: 
        pred_obj = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
        
        # Compute distances to all reference points  
        dists = [np.linalg.norm(pred_obj - ref_point) for ref_point in ref_points]
            
        # Get k nearest
       nearest_dists = sorted(dists)[:k] 
        
        # Mean of the k nearest distances as coverage gap score 
        cov_gap_score = np.mean(nearest_dists)
        
        # Blend with acquisition value (acq_value_norm is already normalized to [0,1])
        acq_val = cand["acq_value_norm"]
        blended_score = acq_val + 0.2 * cov_gap_score
        
        scores.append(blended_score)

    return scores