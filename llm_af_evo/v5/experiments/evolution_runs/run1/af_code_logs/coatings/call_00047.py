def score_pool(context):
    """Score candidates based on acquisition value blended with a coverage-gap term targeting sparse Pareto front regions."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for k-nearest neighbors
    pf = context.get("pareto_front", None)
    if pf is not None and len(pf) >= 3:
        ref_points = pf.copy()
    else:
        ref_points = context['Y_obs'].copy()

    # Compute mean of predicted objectives per candidate 
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        pred_obj_vector = np.array([gp[name]["mean"] for name in names])
      
        distances = [np.linalg.norm(pred_obj_vector - ref_point) for ref_point in ref_points]
        k_nearest_distances = sorted(distances)[:3]  # Take the three nearest
          
        coverage_gap_score = sum(k_nearest_distances) / len(k_nearest_distances)
        
        blended_acq_value_with_coverage = cand["acq_value_norm"] + (0.1 * coverage_gap_score)

        scores.append(blended_acq_value_with_coverage)
    
    return scores