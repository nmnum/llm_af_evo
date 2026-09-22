def score_pool(context):
    """Reward candidates predicted to land in under-covered regions of the Pareto front by measuring mean distance to nearest front points."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Prepare objective vectors for distance computation
    if len(context["pareto_front"]) >= 3:
        pf_points = context["pareto_front"] 
    else:
        pf_points = context["Y_obs"]

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Get candidate's predicted objective vector
        pred_obj = np.array([gp[name]["mean"] for name in names])
        
        if len(pf_points) == 0:
            coverage_score = 0.0  
        else: 
            # Compute distances to all front points and take mean of k=3 nearest
            dists = [np.linalg.norm(pred_obj - pf_point, ord=2) for pf_point in pf_points]
            
            sorted_dists = np.sort(dists)
            if len(sorted_dists) < 3:
                k_nearest = sorted_dists  
            else: 
                k_nearest = sorted_dists[:3] 

            coverage_score = float(np.mean(k_nearest))
        
        # Blend with acquisition value (as a small secondary term added to it).
        blended_score = cand["acq_value_norm"] + 0.1 * coverage_score
        scores.append(blended_score)
    
    return scores