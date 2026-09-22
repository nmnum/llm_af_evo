def score_pool(context):
    """Score candidates based on how much they improve coverage of under-covered regions of the Pareto front."""
    import numpy as np

    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for k=3 nearest neighbors
    pf_points = context['pareto_front']
    if len(pf_points) < 3:
        ref_points = context['Y_obs']  
    else:
        ref_points = pf_points
    
    scores = []
    for cand in context["pool"]:
        # Get candidate's predicted objectives (already maximized)
        pred_obj = np.array([cand["gp_posterior"][name]["mean"] 
                            for name in names])
        
        if len(ref_points) == 0:
            coverage_score = 0.0
        else:
            distances = [np.linalg.norm(pred_obj - ref_point)  
                        for ref_point in ref_points]
            
            # Get k=3 nearest neighbors (or all points, whichever is fewer)
            sorted_dists = np.sort(distances)[:min(3, len(distances))]
            coverage_score = 1.0 / (1e-8 +np.mean(sorted_dists))
        
        # Blend with acquisition value
        acq_value_norm = cand["acq_value_norm"]
        final_score = acq_value_norm + 0.2 * coverage_score
        
        scores.append(final_score)
    
    return scores