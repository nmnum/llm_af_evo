def score_pool(context):
    """Blend acquisition value with uncertainty, and penalize similarity to already observed points."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    x_obs = context["X_obs"]
    
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        
        # Uncertainty bonus: sum of normalized stds
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                        for name in names)
        
        # Novelty penalty: inverse distance to nearest observed point  
        x_cand = cand["x"]
        if len(x_obs) == 0:
            novelty_penalty = 0.0
        else:
            distances = np.linalg.norm(x_obs - x_cand, axis=1)
            min_distance = np.min(distances)
            # Invert distance to get penalty (closer = more penalized)
            novelty_penalty = 1.0 / (min_distance + 1e-8) if min_distance > 0 else 100.0
        
        scores.append(acq + 0.3 * sigma_sum - 0.25 * novelty_penalty)

    return scores