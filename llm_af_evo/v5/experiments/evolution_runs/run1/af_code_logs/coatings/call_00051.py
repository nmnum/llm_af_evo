def score_pool(context):
    """Blend acquisition value with a coverage-gap score based on distance to nearest Pareto front points."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto_front is too small for meaningful density estimation
    use_y_obs = len(pf) < 3
    
    scores = []
    for cand in context["pool"]:
        gp_means = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to nearest front points (or Y_obs if needed)
        if use_y_obs:
            obs_points = context["Y_obs"]
        else:
            obs_points = pf
            
        dists_to_front = []
        k_nearest = min(3, len(obs_points))
        
        for point in obs_points:
            distance_sq = np.sum((gp_means - point) ** 2)
            dists_to_front.append(distance_sq)

        # Get mean of squared distances to the nearest points
        sorted_dists = sorted(dists_to_front)[:k_nearest]
        coverage_gap_score = sum(sorted_dists) / k_nearest
        
        blended_score = cand["acq_value_norm"] + 0.1 * np.sqrt(coverage_gap_score)
        
        scores.append(blended_score)

    return scores