def score_pool(context):
    """Score candidates based on how much they expand under-covered regions of the Pareto front by incorporating a coverage-gap term into the acquisition value."""
    import numpy as np
    
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto_front is too small
    use_pf = len(pf) >= 3
    ref_points = context["Y_obs"] if not use_pf else pf

    scores = []
    for cand in context["pool"]:
        gp_means = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to k nearest points (k=3)
        dists = [np.linalg.norm(gp_means - ref_point) for ref_point in ref_points]
        dists.sort()
        mean_dist_to_k_nearest = np.mean(dists[:min(3, len(ref_points))])

        # Invert the distance: smaller distances to front neighbors means more covered
        coverage_gap_score = 1.0 / (mean_dist_to_k_nearest + 1e-8)
        
        blended_score = cand["acq_value_norm"] + 0.2 * coverage_gap_score  
        scores.append(blended_score)

    return scores