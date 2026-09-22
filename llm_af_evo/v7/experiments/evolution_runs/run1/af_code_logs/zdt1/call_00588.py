def modifier(context):
    """Bonus for candidates near under-covered regions of Pareto front combined with penalty for similarity to already-selected candidates."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    y_min = np.array([context["ref_point_by_name"][name] for name in names])
    y_max = np.array([max(context["Y_obs"][:, i]) for i, name in enumerate(names)])
    ranges = y_max - y_min
    
    values = []
    
    # Track selected candidates to penalize similarity
    selected_x = []

    progress_factor = 1.0 - context["campaign"]["progress"]
    
    for cand in context["pool"]:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        norm_mean = (mean_vec - y_min) / ranges
        
        distances = np.linalg.norm((context["Y_obs"] - y_min) / ranges - norm_mean, axis=1)
        min_distance = np.min(distances)

        acq_value_norm = cand["acq_value_norm"]
        
        # Front coverage bonus
        front_bonus = min(min_distance, 0.3) * acq_value_norm
        
        # Similarity penalty (based on x-space distance to already selected)
        cand_x = cand["x"]
        min_selected_dist_sq = float('inf')
        for prev_x in selected_x:
            dist_sq = np.sum((cand_x - prev_x) ** 2)
            if dist_sq < min_selected_dist_sq:
                min_selected_dist_sq = dist_sq
        
        distance = np.sqrt(min_selected_dist_sq) if min_selected_dist_sq != float('inf') else 0.0
        similarity_penalty_multiplier = 1.0 - np.exp(-distance * 3.0)
        
        # Combine bonuses and penalties (penalty is negative, so subtract it)
        combined_value = front_bonus + ((similarity_penalty_multiplier - 1.0) * 0.25 * progress_factor)

        values.append(combined_value)
        
        selected_x.append(cand_x)

    return values