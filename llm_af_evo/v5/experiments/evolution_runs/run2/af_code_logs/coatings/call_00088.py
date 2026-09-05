def score_pool(context):
    """Score candidates based on acquisition value blended with a coverage-gap term targeting under-covered regions of the Pareto front."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for k=3 nearest neighbors
    pf = context.get("pareto_front", None)
    if pf is not None and len(pf) >= 3:
        points_for_nn = pf.copy()
    else:
        points_for_nn = context["Y_obs"].copy()

    scores = []
    
    # Precompute distances for all candidates to the reference set
    k_nearest = min(3, len(points_for_nn))
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        pred_obj_vector = np.array([gp[name]["mean"] for name in names])
        
        if points_for_nn.shape[0] == 1:
            dists_to_front = [np.linalg.norm(pred_obj_vector - points_for_nn[0])]
        else:
            # Compute distances from this candidate to all front/observed points
            diffs = pred_obj_vector[np.newaxis, :] - points_for_nn 
            dists_to_front = np.sqrt(np.sum(diffs**2, axis=1)).tolist()
            
        sorted_dists = sorted(dists_to_front)
        
        mean_dist_knn = sum(sorted_dists[:k_nearest]) / k_nearest
        
        # Use acquisition value as base score and add coverage gap term
        acq_score = cand["acq_value_norm"]
        blended_score = acq_score + 0.1 * (mean_dist_knn) 
        
        scores.append(blended_score)
        
    return scores