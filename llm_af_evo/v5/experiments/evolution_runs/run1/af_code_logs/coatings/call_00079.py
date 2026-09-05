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
        
        # Compute distances to front points (or Y_obs if too few)
        ref_points = pf if not use_yobs else context["Y_obs"]
        k_nearest_dists = []
        for i, pt in enumerate(ref_points):
            dist_sq = sum((pred_obj[j] - pt[j])**2 for j in range(len(names)))
            k_nearest_dists.append(dist_sq)
        
        # Get the 3 smallest distances (or all if fewer than 3) 
        sorted_dist_indices = np.argsort(k_nearest_dists)[:min(3, len(ref_points))]
        mean_knn_distance = sum(np.sqrt(k_nearest_dists[i]) for i in sorted_dist_indices)/len(sorted_dist_indices)
        
        # Blend with acquisition value: higher acq_value + high coverage-gap score
        blended_score = cand["acq_value_norm"] + 0.2 * (1 - mean_knn_distance) 
        scores.append(blended_score)

    return scores