def score_pool(context):
    """Estimate improvement potential using hypervolume contribution uncertainty-aware scoring."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted mean and std per objective
        mu = [gp[name]["mean"] for name in names]
        sigma = [gp[name]["std"] for name in names]

        # Compute hypervolume contribution (approximated by distance from ref point)
        hv_contribution = np.prod(np.maximum(ref_point - mu, 0))

        # Add uncertainty penalty: higher std means more uncertain predictions
        ucb_penalty = sum(sigma[i] / front_range[name] for i, name in enumerate(names))
        
        scores.append(hv_contribution * (1.0 + 0.5 * ucb_penalty)) 

    return scores