def modifier(context):
    """Adaptive uncertainty bonus scaled by how close a candidate is to existing observations, with suppression of near-duplicates."""
    if len(context["X_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Normalize observations and candidates
    x_min = np.min(context["X_obs"], axis=0)
    x_max = np.max(context["X_obs"], axis=0)  
    x_ranges = x_max - x_min
    
    normalized_X_obs = (context["X_obs"] - x_min) / x_ranges if not np.all(x_ranges == 0) else context["X_obs"]
    
    values = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Compute uncertainty bonus
        sigma_norm = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        
        # Measure proximity to existing observations  
        normalized_x_cand = (cand["x"] - x_min) / x_ranges if not np.all(x_ranges == 0) else cand["x"]
        distances_to_obs = np.linalg.norm(normalized_X_obs - normalized_x_cand, axis=1)
        
        # Suppression factor based on minimum distance to any observed point
        min_dist = np.min(distances_to_obs)
        suppression_factor = max(0.5, 1.0 / (min_dist + 0.1)) if min_dist > 0 else 2.0
        
        base_weight = 0.3 * (1.0 - context["campaign"]["progress"])
        
        # Scale by stagnation level for adaptive exploration
        scaling_factor = max(0.5, 1.0 - stagnant_batches / 10.)
        weight = base_weight * scaling_factor

        values.append(weight * sigma_norm * suppression_factor)

    return values