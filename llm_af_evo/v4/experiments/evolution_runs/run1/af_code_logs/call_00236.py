def score_pool(context):
    """Incorporate acquisition value saturation with uncertainty-driven preference for diverse front regions."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"] 
        gp_posterior = cand["gp_posterior"]

        # Normalize candidate's predicted objectives
        pred_objs = np.array([gp_posterior[name]["mean"] for name in names])  
        
        # Compute hypervolume contribution (simplified proxy)
        hv_contribution = 1.0 - np.prod(ref_point - pred_objs) / np.prod(ref_point)

        # Add uncertainty bonus, scaled by how far we are from the current front
        sigma_sum = sum(gp_posterior[name]["std"] for name in names)
        
        # Use a sigmoid to blend acquisition and hypervolume contribution 
        scores.append(acq * (0.5 + 0.5 * hv_contribution) + 0.1 * sigma_sum)

    return scores