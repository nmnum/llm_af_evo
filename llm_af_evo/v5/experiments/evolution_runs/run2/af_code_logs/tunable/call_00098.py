def score_pool(context):
    """Blend acquisition value with a coverage-gap score based on distance to nearest Pareto front points."""
    names = context["objective_names"]
    pf = context["pareto_front"] 
    acq_norm = [cand['acq_value_norm'] for cand in context["pool"]]
    
    if len(pf) < 3:
        # Use all observations as reference when front is too small
        ref_points = context["Y_obs"]
    else:
        ref_points = pf
    
    scores = []
    k = min(3, len(ref_points))
    
    for cand in context["pool"]:
        gp_means = np.array([cand['gp_posterior'][name]["mean"] for name in names])
        
        # Compute distances to nearest reference points
        dists = [np.linalg.norm(gp_means - ref_point) for ref_point in ref_points]
        dists.sort()
        mean_dist_to_front = sum(dists[:k]) / k
        
        # Invert so sparse areas get higher scores (lower distance means more coverage)
        inverse_coverage_gap_score = 1.0 / (mean_dist_to_front + 1e-8) 
        
        blended_score = acq_norm[context["pool"].index(cand)] + 0.2 * inverse_coverage_gap_score
        scores.append(blended_score)

    return scores