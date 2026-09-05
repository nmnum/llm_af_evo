def modifier(context):
    """Novelty bonus scaled by acquisition strength: rewards candidates far from observed points, but only when they already have high acquisition value."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    acq_values = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Compute distances to nearest observations
    x_obs = context["X_obs"]
    x_pool = np.array([cand["x"] for cand in context["pool"]])
    min_distances = []
    for i, x_cand in enumerate(x_pool):
        if len(x_obs) == 0:
            dist = float("inf")
        else:
            distances = np.linalg.norm(x_obs - x_cand, axis=1)
            dist = np.min(distances)
        min_distances.append(dist)

    # Normalize distance by the range of features
    feature_range = np.max(context["X_obs"], axis=0) - np.min(context["X_obs"], axis=0)
    if not np.all(feature_range > 0):
        normalized_dists = [1.0] * len(min_distances)
    else:
        # Avoid division by zero; use a small epsilon for normalization
        eps = 1e-8 
        feature_range[feature_range == 0] = eps  
        normalized_dists = np.array([d / r if d != float("inf") and not (r <=eps)else 1.0 for d, r in zip(min_distances, feature_range)])

    # Scale bonus by acquisition strength: only apply to strong candidates
    threshold_acq = max(acq_values) * 0.5  
    values = []
    for i, cand in enumerate(context["pool"]):
        if acq_values[i] >= threshold_acq:
            weight_novelty = (1 - normalized_dists[i]) 
            gp = cand["gp_posterior"]
            sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names)
            values.append(0.3 * weight_novelty * sigma_sum)  
        else:  # Low acquisition candidates get no bonus
            values.append(0.)
    return values