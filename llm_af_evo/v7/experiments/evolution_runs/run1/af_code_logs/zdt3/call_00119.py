def modifier(context):
    """Bonus for candidates near under-covered regions of the Pareto front, with novelty penalty for already-selected batch members."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    y_min = np.array([context["ref_point_by_name"][name] for name in names])
    y_max = np.array([max(context["Y_obs"][:, i]) for i, name in enumerate(names)])
    ranges = y_max - y_min
    normalized_y_obs = (context["Y_obs"] - y_min) / ranges
    
    values = []
    progress_factor = 1.0 - context["campaign"]["progress"]
    
    # Track selected candidates to penalize duplicates
    selected_x = []

    for i, cand in enumerate(context["pool"]):
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        norm_mean = (mean_vec - y_min) / ranges
        
        distances = np.linalg.norm(normalized_y_obs - norm_mean, axis=1)
        min_distance = np.min(distances)

        acq_value_norm = cand["acq_value_norm"]
        
        # Front coverage bonus
        scaled_bonus = min(min_distance, 0.3) * acq_value_norm * progress_factor
        
        # Duplicate penalty (only applies if candidate has been selected already)
        dup_penalty = 0.0
        for prev_x in selected_x:
            dist_sq = np.sum((cand["x"] - prev_x) ** 2)
            distance = np.sqrt(dist_sq)
            multiplier = 1.0 - np.exp(-distance)
            # Apply penalty if close to any previously chosen candidate (penalty scaled by acquisition value and progress factor)
            dup_penalty += max(0, (multiplier - 1.0)) * acq_value_norm
        
        final_bonus = scaled_bonus + dup_penalty
        values.append(final_bonus)

        selected_x.append(cand["x"])

    return values