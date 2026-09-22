def score_pool(context):
    """Score candidates based on how much they extend under-covered regions of the Pareto front."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for k=3 nearest neighbors
    pf = context['pareto_front']
    use_yobs = len(pf) < 3
    ref_points = context['Y_obs'] if use_yobs else pf
    
    scores = []
    for cand in context["pool"]:
        # Get candidate's predicted objectives (already maximized)
        pred_obj = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to k=3 nearest points
        if len(ref_points) == 0:
            scores.append(cand['acq_value_norm'])
            continue
            
        dists_to_ref = [np.linalg.norm(pred_obj - ref_point, ord=2)
                        for ref_point in ref_points]
        sorted_dists = np.sort(dists_to_ref)[:3] if len(ref_points) >= 3 else sorted_dists
        
        # Mean of k nearest distances
        mean_dist = np.mean(sorted_dists) if len(sorted_dists) > 0 else float('inf')
        
        # Invert to score higher for sparse areas (low coverage)
        inverse_coverage_score = -mean_dist + max(dists_to_ref) * 2.0
        
        blended_score = cand['acq_value_norm'] + 0.15 * inverse_coverage_score
        scores.append(blended_score)

    return scores