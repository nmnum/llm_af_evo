def score_pool(context):
    """Score candidates based on how much they expand under-covered regions of the Pareto front."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for k=3 nearest neighbors
    pf = context['pareto_front']
    obs = context['Y_obs'] 
    use_pf = len(pf) >= 3
    ref_points = pf if use_pf else obs
    
    scores = []
    for cand in context["pool"]:
        # Get candidate's predicted objectives (already maximized)
        mu_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to k=3 nearest front points
        dists = [np.linalg.norm(mu_vec - ref_pt) for ref_pt in ref_points]
        dists.sort()
        knn_dists = dists[:min(3, len(dists))]
        
        coverage_gap_score = np.mean(knn_dists)
        # Blend with acquisition value (add as secondary term)
        score = cand["acq_value_norm"] + 0.1 * coverage_gap_score
        scores.append(score)

    return scores