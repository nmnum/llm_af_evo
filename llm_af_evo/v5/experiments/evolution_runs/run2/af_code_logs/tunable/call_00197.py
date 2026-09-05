def score_pool(context):
    """Score candidates based on how much they extend under-covered regions of the Pareto front by incorporating a coverage-gap term derived from nearest-front-point distances."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for meaningful density estimation
    points_for_dist = context.get("pareto_front", [])
    if len(points_for_dist) < 3:
        points_for_dist = context["Y_obs"]

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted objective vector (already maximized)
        pred_obj = np.array([gp[name]["mean"] for name in names])
    
        if len(points_for_dist) < 1:
            coverage_gap_score = 0.0
        else:
            distances_to_front = [
                np.linalg.norm(pred_obj - point, ord=2)
                for point in points_for_dist[:len(points_for_dist)]
            ]
            
            # Get mean of k nearest (k=3 or fewer if less than three exist) 
            sorted_distances =sorted(distances_to_front)[:min(3,len(distances_to_front))]
            coverage_gap_score = np.mean(sorted_distances)
        
        blended_acq_value = cand["acq_value_norm"] + 0.1 * coverage_gap_score
        scores.append(blended_acq_value)

    return scores