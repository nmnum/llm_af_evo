def score_pool(context):
    """Blend acquisition value with a coverage-gap score based on distance to nearest Pareto front points."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto_front is too small for k=3 neighbors
    use_yobs = len(pf) < 3
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        pred_obj = np.array([gp[name]["mean"] for name in names])
        
        # Compute distances to nearest front points (or Y_obs if needed)
        if use_yobs:
            obs_points = context["Y_obs"]
        else:
            obs_points = pf
            
        dists_to_front = []
        for point in obs_points: 
            d = np.linalg.norm(pred_obj - point)  # L2 distance
            dists_to_front.append(d)

        k_nearest_dists = sorted(dists_to_front)[:3]
        coverage_gap_score = sum(k_nearest_dists)/len(k_nearest_dists)
        
        blended_score = cand["acq_value_norm"] + (0.1 * coverage_gap_score)  # small secondary term
        scores.append(blended_score)

    return scores