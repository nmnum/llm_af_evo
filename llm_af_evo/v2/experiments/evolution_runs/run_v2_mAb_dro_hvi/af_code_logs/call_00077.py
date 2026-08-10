def score_pool(context):
    """Estimate improvement potential using hypervolume expansion minus novelty penalty."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Score each candidate based on predicted HV contribution and distance from observed points
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        pred_means = np.array([gp[name]["mean"] for name in names])  
        
        # Estimate hypervolume improvement (simplified proxy)
        hv_improvement = 1.0 - np.prod(np.maximum(ref_point - pred_means, 0) / ref_point)

        # Add novelty bonus: inverse distance to nearest observed point
        cand_x = cand["x"]
        if len(context["X_obs"]) > 0:
            distances = np.linalg.norm(context["X_obs"] - cand_x, axis=1)
            min_distance = np.min(distances) 
            novel_bonus = 1.0 / (min_distance + 1e-8) # avoid division by zero
        else:  
            novel_bonus = 1.0
            
        scores.append(hv_improvement * novel_bonus)

    return scores