def score_pool(context):
    """Invert uncertainty scaling based on acquisition value to boost low-acq candidates with high confidence."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
        
        # When acquisition is low, boost candidates with lower uncertainty
        if acq < 0.5:
            scale_factor = 1.0 + (0.3 * (1.0 - sigma_sum))
        else:
            # For high-acq candidates, reduce the effect of uncertainty to preserve exploitation signal
            scale_factor = 1.0 - (0.2 * sigma_sum)
            
        scores.append(acq * scale_factor)

    return scores