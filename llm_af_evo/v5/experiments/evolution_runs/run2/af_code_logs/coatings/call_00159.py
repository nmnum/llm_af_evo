def score_pool(context):
    """Blend acquisition value with an uncertainty-adjusted frontier proximity signal that rewards candidates near sparse frontiers but penalizes overconfident predictions in crowded regions."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    ref_point = context["ref_point"]
    
    # Compute candidate distances to nearest Pareto points
    if len(pf) < 1:
        min_distances = np.full(len(context["pool"]), float('inf'))
    else:
        candidates_means = np.array([list(cand["gp_posterior"][name]["mean"] for name in names)
                                    for cand in context["pool"]])
        # Use sklearn's pairwise distances if available, otherwise fallback to manual
        try:
            from scipy.spatial.distance import cdist
            min_distances = np.min(
                cdist(candidates_means, pf), axis=1
            )
        except ImportError:
            # Manual computation for compatibility without scipy (slower but works)
            def point_to_front_dist(point):
                return min(np.linalg.norm(np.array(point) - np.array(front_point)) 
                          for front_point in pf)

            min_distances = [point_to_front_dist(mean_vec) for mean_vec in candidates_means]

    # Normalize distances to 0-1 scale based on observed ranges
    range_vals = list(context["pareto_front_range"][name] for name in names)
    max_distance = np.sqrt(sum(r**2 for r in range_vals)) if len(range_vals) > 0 else 1.0
    
    scores = []
    
    # For each candidate, compute uncertainty-adjusted proximity score
    acq_values = [cand["acq_value_norm"] for cand in context["pool"]]
    
    for i, (cand, dist_to_front) in enumerate(zip(context["pool"], min_distances)):
        gp_posterior = cand["gp_posterior"]
        
        # Normalized uncertainty: sum of normalized standard deviations
        sigma_sum_normalized = np.sum([
            gp_posterior[name]["std"] / context["pareto_front_range"][name]
            for name in names])
            
        # Normalize the distance to [0, 1] (smaller is better)
        dist_normed = min(1.0, max_distance and dist_to_front/max_distance or 0.)
        
        if sigma_sum_normalized > 0:
            unct_penalty = np.exp(-sigma_sum_normalized) * (
                -dist_normed + 1
            )
        else:
            # No uncertainty penalty when std is zero (e.g., near observed points)
            unct_penalty = -(dist_normed)

        score_i = acq_values[i] + .5*unct_penalty
        
        scores.append(score_i) 
    
    return scores