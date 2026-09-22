def score_pool(context):
    """Balance acquisition value with uncertainty-adjusted progress toward frontier expansion."""
    if not context["pool"]:
        return []
    
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    front_range = context["pareto_front_range"]

    # Compute base scores using acquisition values and uncertainty
    acq_values = [cand["acq_value_norm"] for cand in context["pool"]]
    
    uncertainties = []
    for cand in context["pool"]:
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                        for name in names)
        uncertainties.append(sigma_sum)

    # Normalize uncertainty to [0, 1]
    max_uncertainty = max(uncertainties) if any(u > 0 for u in uncertainties) else 1.0
    normalized_uncerts = [u / max_uncertainty for u in uncertainties]

    scores = []
    for i, (acq_val, norm_u) in enumerate(zip(acq_values, normalized_uncerts)):
        # Use uncertainty to modulate the acquisition value dynamically based on progress:
        campaign_progress = context["campaign"]["progress"]
        
        # Early exploration: emphasize higher uncertainty
        if campaign_progress < 0.3:
            weight_factor = (1 - norm_u) * 2   # prefer less uncertain when early, but not too much  
        else:
            # Later exploitation: reduce emphasis on uncertainty 
            weight_factor = max(0., 1.5*(1-norm_u)) 
            
        score = acq_val + weight_factor * min(norm_u, 0.8)
        
        scores.append(score)

    return [float(s) for s in scores]