def score_pool(context):
    """Score candidates based on how well their predicted objectives align with ref point in normalized space, penalizing overconfident predictions."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    front_range = context["pareto_front_range"]

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize predicted means to [0, 1] relative to ref point and front range
        normalized_means = np.array([
            (ref_point[i] - gp[name]["mean"]) / front_range[name]
            if not np.isclose(front_range[name], 0) else 0.5 
            for i, name in enumerate(names)
        ])
        
        # Compute squared Mahalanobis-like distance from ref point
        inv_ranges = np.array([1./front_range[name] if not np.isclose(front_range[name], 0.) else 1. for name in names])
        uncertainty_penalty = sum((gp[name]["std"] * inv_ranges[i])**2 
                                  for i, name in enumerate(names))
        
        # Score is the inverse of distance to reference point adjusted by prediction confidence
        score = np.exp(-np.sum(normalized_means**2) / 2.0 - uncertainty_penalty)
        scores.append(score)

    return scores