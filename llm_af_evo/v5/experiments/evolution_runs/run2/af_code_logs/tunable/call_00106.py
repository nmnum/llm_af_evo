def score_pool(context):
    """Blend acquisition value with a coverage-gap score based on distance to nearest front points."""
    import numpy as np
    
    names = context["objective_names"]
    pf = context["pareto_front"] 
    acq_values = [cand["acq_value_norm"] for cand in context["pool"]]
    
    if len(pf) < 3:
        ref_points = context["Y_obs"]
    else:
        ref_points = pf
    
    scores = []
    k = 3
    for cand in context["pool"]:
        pred_obj = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        distances = [np.linalg.norm(pred_obj - point) for point in ref_points]
        nearest_dists = sorted(distances)[:k] 
        coverage_gap_score = sum(nearest_dists) / k
        
        blended score = acq_values[i] + 0.1 * coverage_gap_score
        scores.append(blended_score)
        
    return scores