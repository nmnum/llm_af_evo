def score_pool(context):
    """Scale acquisition values by uncertainty inversely to encourage exploration of less confident regions while preserving hypervolume signal."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
        
        # Inverse scaling: lower uncertainty -> higher score boost
        if sigma_sum > 0:
            scale_factor = 1.0 + (0.5 * (1.0 - sigma_sum))  
        else:
            scale_factor = 1.0
            
        scores.append(acq * scale_factor)
    
    return scores