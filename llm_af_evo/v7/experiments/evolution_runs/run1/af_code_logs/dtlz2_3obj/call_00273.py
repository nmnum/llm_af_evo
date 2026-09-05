def modifier(context):
    """Penalize candidates that are close in objective space to already observed points, encouraging exploration of under-covered regions."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    y_min = ref_point
    y_max = np.array([max(context["Y_obs"][:, i]) for i, name in enumerate(names)])
    ranges = y_max - y_min
    
    # Normalize observations and candidates' means
    normalized_y_obs = (context["Y_obs"] - y_min) / ranges if not np.all(ranges == 0) else context["Y_obs"]
    
    values = []
    progress_factor = 1.0 - context["campaign"]["progress"]

    for cand in context["pool"]:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        norm_mean = (mean_vec - y_min) / ranges if not np.all(ranges == 0) else mean_vec
        
        # Compute distances to all observed points
        distances = []
        try:
            dist_matrix = np.linalg.norm(normalized_y_obs[:, None] - norm_mean[None, :], axis=2)
            min_distance = np.min(dist_matrix).item()
        except Exception:
            min_distance = float('inf')
        
        # Compute bonus: candidates near uncovered regions get a higher score
        acq_value_norm = cand["acq_value_norm"]
        front_bonus = (1.0 - min_distance) * 0.3 if not np.isinf(min_distance) else 0.0
        
        values.append(front_bonus)

    return values