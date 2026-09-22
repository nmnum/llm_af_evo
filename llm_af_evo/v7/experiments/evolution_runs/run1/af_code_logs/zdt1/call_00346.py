def modifier(context):
    """Penalty for candidates that are too similar to already-observed points, encouraging diversity in selected batch."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    y_min = np.array([context["ref_point_by_name"][name] for name in names])
    y_max = np.array([max(context["Y_obs"][:, i]) for i, name in enumerate(names)])
    ranges = y_max - y_min
    
    # Normalize observations and candidates
    normalized_y_obs = (context["Y_obs"] - y_min) / ranges

    values = []
    for cand in context["pool"]:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        norm_mean = (mean_vec - y_min) / ranges
        
        # Compute distances to all observed points
        distances = np.linalg.norm(normalized_y_obs - norm_mean, axis=1)
        
        # Use the minimum distance as a measure of similarity
        min_distance = np.min(distances)

        # Apply penalty inversely proportional to proximity (higher penalty for closer candidates)  
        acq_value_norm = cand["acq_value_norm"]
        if min_distance < 0.05:
            scaled_penalty = -1.0 * acq_value_norm 
        else:   
            scaled_penalty = -(min_distance / 0.3)**2 * acq_value_norm

        values.append(scaled_penalty)

    return values