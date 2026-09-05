def score_pool(context):
    """Score candidates based on how much they improve coverage gaps in the Pareto front."""
    import numpy as np
    
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto_front is too small for k=3 nearest neighbors
    use_y_obs = len(pf) < 3
    ref_points = context["Y_obs"] if use_y_obs else pf
    
    scores = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to k nearest points (k=3)
        dists = [np.linalg.norm(gp_mean - pt) for pt in ref_points]
        sorted_dists = sorted(dists)[:min(3, len(ref_points))]
        coverage_gap_score = np.mean(sorted_dists) if sorted_dists else 0.0
        
        # Blend with acquisition value (smaller secondary term)
        blended_score = cand["acq_value_norm"] + 0.1 * coverage_gap_score
        scores.append(blended_score)

    return scores