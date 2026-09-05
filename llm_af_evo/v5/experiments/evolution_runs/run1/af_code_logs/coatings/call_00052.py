def score_pool(context):
    """Blend hypervolume acquisition with a coverage-gap term targeting sparse Pareto front regions."""
    import numpy as np
    
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto front is too small for k=3 nearest neighbors
    obs_points = pf if len(pf) >= 3 else context["Y_obs"] 
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        pred_obj = np.array([gp[name]["mean"] for name in names])
                
        # Compute distances to k=3 nearest front points
        dists_to_front = [np.linalg.norm(pred_obj - pt) for pt in obs_points]
        dists_to_front.sort()
        knn_dists = dists_to_front[:min(3, len(dists_to_front))]
        
        coverage_gap_score = np.mean(knn_dists)
                
        # Blend with acquisition value
        final_score = cand["acq_value_norm"] + 0.1 * coverage_gap_score
        
        scores.append(final_score)

    return scores