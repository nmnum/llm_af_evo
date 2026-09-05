def modifier(context):
    """Adaptive uncertainty bonus scaled by stagnation and combined with novelty reward."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Base uncertainty bonus with decay based on stagnation
    weight = 0.3 / (1 + stagnant_batches)
    values = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Uncertainty term: sum of normalized stds 
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names) 
        
        # Novelty bonus based on distance to nearest observed point
        x_cand = cand["x"]
        distances = np.linalg.norm(context["X_obs"] - x_cand, axis=1)
        min_distance = np.min(distances)
        
        novelty_bonus = 0.5 * (min_distance / max(np.max(distances), 1e-8))
        
        # Combine uncertainty and novelty with adaptive weight
        total_correction = weight * sigma_norm + novelty_bonus
        
        values.append(total_correction)

    return values