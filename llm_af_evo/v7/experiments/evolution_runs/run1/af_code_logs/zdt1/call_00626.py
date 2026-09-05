def modifier(context):
    """Adaptive uncertainty bonus scaled by acquisition strength and stagnation level."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    campaign = context["campaign"]
    
    # Base weight decays with progress, increases with staleness  
    base_weight = 0.3 * (1.0 - campaign["progress"]) + 0.2 * min(5, campaign["stagnant_batches"])
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Compute normalized uncertainty across objectives
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Scale bonus by how acquisition-strong the candidate already is  
        acq_weight = 1.0 + (cand["acq_value_norm"] - 0.5) * 2.0
        
        values.append(base_weight * sigma_norm * acq_weight)

    return values