def modifier(context):
    """Combine objective-space novelty bonus with diversity penalty against selected candidates."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    y_min = np.array([context["ref_point_by_name"][name] for name in names])
    y_max = np.array([max(context["Y_obs"][:, i]) for i, name in enumerate(names)])
    ranges = y_max - y_min
    normalized_y_obs = (context["Y_obs"] - y_min) / ranges
    
    values = []
    selected_x = []  # Track already-selected candidates' x-values to penalize duplicates

    for cand in context["pool"]:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        norm_mean = (mean_vec - y_min) / ranges
        
        distances = np.linalg.norm(normalized_y_obs - norm_mean, axis=1)
        min_distance = np.min(distances)
        
        # Novelty bonus: higher when farther from existing observations
        novelty_bonus = min_distance * 0.5

        cand_x = cand["x"]
        # Diversity penalty against already-selected candidates (in x-space)
        min_x_dist_sq = float('inf')
        for prev_x in selected_x:
            dist_sq = np.sum((cand_x - prev_x) ** 2)
            if dist_sq < min_x_dist_sq:
                min_x_dist_sq = dist_sq
        
        distance = np.sqrt(min_x_dist_sq) if min_x_dist_sq != float('inf') else 0.01
        penalty_multiplier = 1.0 - np.exp(-distance * 5.) # Scale up the x-space sensitivity

        diversity_penalty = (penalty_multiplier - 1.0) * 0.244
        
        total_correction = novelty_bonus + diversity_penalty
        values.append(total_correction)
        
        selected_x.append(cand_x)

    return values