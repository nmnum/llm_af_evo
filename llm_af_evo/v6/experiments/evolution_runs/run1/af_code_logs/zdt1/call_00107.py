def modifier(context):
    """Reward candidates with high uncertainty and low acquisition value to encourage exploration of under-investigated regions."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    values = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Normalize uncertainty across objectives
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                        for name in names)
                        
        # If acquisition value is low but uncertainty is high, apply a bonus (to encourage exploration of uncertain regions)  
        if cand["acq_value_norm"] < 0.3 and sigma_sum > 0.5:
            values.append(sigma_sum * 0.2) 
        else: 
            values.append(0.)

    return values