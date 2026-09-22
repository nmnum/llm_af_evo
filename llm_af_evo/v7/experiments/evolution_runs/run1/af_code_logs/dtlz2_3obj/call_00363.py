def modifier(context):
    """Penalty for candidates with low uncertainty when acquisition value is high, encouraging exploration of promising yet uncertain regions."""
    names = context["objective_names"]
    values = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Compute total predictive standard deviation across objectives
        sigma_total = sum(gp[name]["std"] for name in names)
        
        acq_value_norm = cand["acq_value_norm"]

        # Apply a penalty when candidate is both highly acquisition-scored and certain (low uncertainty),
        # encouraging exploration of promising uncertain candidates instead.
        if acq_value_norm > 0.5: 
            correction = -sigma_total * 0.1
        else:
            correction = 0.0
            
        values.append(correction)
        
    return values