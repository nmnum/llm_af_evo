def modifier(context):
    """Combine uncertainty bonus with objective-space novelty, scaling novelty by acquisition strength."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    # Base uncertainty bonus (UCB-style)
    weight_u = 0.3 * (1.0 - progress) 
    values_uncertainty = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]        
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        values_uncertainty.append(weight_u * sigma_norm)

    # Novelty bonus based on objective space distance
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
        values_novelty.append(min_distance * 0.5)

    # Normalize novelty bonus to [0, 1] range and scale by acquisition value strength  
    max_val = max(values_novelty) if any(v > 0 for v in values_novelty) else 1.0
    normalized_novelties = [v / max_val * 0.5 for v in values_novelty]
    
    # Apply a multiplicative factor to novelty based on how strong the acquisition value is  
    acq_values = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    mean_acq = np.mean(acq_values)
    scale_factors = 1.0 + (acq_values - mean_acq) * 2.0
    
    final_novelties = [novelty * factor for novelty, factor in zip(normalized_novelties, scale_factors)]
    
    # Combine uncertainty and scaled novelty bonuses
    return [(u_bonus + n_bonus) for u_bonus, n_bonus in zip(values_uncertainty, final_novelties)]