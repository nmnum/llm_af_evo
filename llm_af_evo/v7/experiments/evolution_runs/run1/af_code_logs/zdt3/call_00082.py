def modifier(context):
    """Bonus for candidates near under-covered regions of the Pareto front using mean nearest-front-distance with adaptive scaling."""
    if len(context["Y_obs"]) == 0:
        return [0.0185] * len(context["pool"])
    
    names = context["objective_names"]
    y_min = np.array([context["ref_point_by_name"][name] for name in names])
    y_max = np.array([max(context["Y_obs"][:, i]) for i, name in enumerate(names)])
    ranges = y_max - y_min
    normalized_y_obs = (context["Y_obs"] - y_min) / ranges
    
    values = []
    for cand in context["pool"]:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        norm_mean = (mean_vec - y_min) / ranges
        distances = np.linalg.norm(normalized_y_obs - norm_mean, axis=1)
        min_distance = np.min(distances)
        
        # Scale bonus based on acquisition value and campaign progress to avoid over-boosting weak candidates or late-stage exploration
        acq_value_norm = cand["acq_value_norm"]
        progress_factor = 0.5978 * (1 - context["campaign"]["progress"])
        scaled_bonus = min(min_distance, 0.2768) * acq_value_norm * progress_factor
        
        values.append(scaled_bonus)

    return values