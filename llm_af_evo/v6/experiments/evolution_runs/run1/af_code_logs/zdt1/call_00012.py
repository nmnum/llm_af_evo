def modifier(context):
    """Combines uncertainty bonus with novelty reward; scales uncertainty bonus by campaign progress and novel candidates get higher bonuses."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    weight_uncertainty = 0.3 * (1.0 - progress)
    
    # Novelty computation
    y_min = np.array([context["ref_point_by_name"][name] for name in names])
    y_max = np.array([max(context["Y_obs"][:, i]) for i, name in enumerate(names)])
    ranges = y_max - y_min
    
    normalized_y_obs = (context["Y_obs"] - y_min) / ranges
    values_novelty = []
    
    for cand in context["pool"]:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        norm_mean = (mean_vec - y_min) / ranges
        
        distances = np.linalg.norm(normalized_y_obs - norm_mean, axis=1)
        min_distance = np.min(distances)
        
        # Normalize by max distance to avoid over-weighting
        if len(context["Y_obs"]) > 0:
            normalized_dist = min_distance * (2.0 / np.sqrt(len(names)))
        else:
            normalized_dist = 0
        
        values_novelty.append(normalized_dist)

    # Combine uncertainty and novelty bonuses, scaled by acquisition value strength  
    max_val = max(values_novelty) if any(v > 0 for v in values_novelty) else 1.0
    final_values = []
    
    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]
        
        # Uncertainty bonus (UCB-style)
        sigma_sum_normed = sum(gp[name]["std"] / front_range[name] for name in names) 
        uncertainty_bonus = weight_uncertainty * sigma_sum_normed
        
        # Novelty-based boost
        novelty_boost = values_novelty[i]
        if max_val > 0:
            normalized_novelty = novelty_boost / max_val  
        else:   
            normalized_novelty = 0.5
            
        combined_bonus = uncertainty_bonus + (normalized_novelty * 0.3)
        
        final_values.append(combined_bonus)

    return final_values