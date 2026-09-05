def modifier(context):
    """Penalty for candidates with low uncertainty and high acquisition value to avoid premature convergence."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    values = []
    
    # Compute a measure that combines exploitation (acq_value_norm) and exploration (uncertainty)
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Normalize uncertainty across objectives
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                        for name in names)

        # If acquisition value is high but uncertainty is low, apply a penalty (to avoid getting stuck)
        if cand["acq_value_norm"] > 0.7 and sigma_sum < 0.3:
            values.append(-sigma_sum * 0.2)  
        else: 
            values.append(0.)

    return values