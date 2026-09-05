def modifier(context):
    """Adaptive uncertainty bonus with stagnation-aware decay and acquisition-strength scaling."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Base weight decays with campaign progression
    base_weight = 0.3 * (1 - progress)
    
    # Increase bonus when stagnating, capped at a maximum boost
    stagnation_boost = min(stagnant_batches / 3.0, 1.0) 
    
    # Combine decay and stagnation effects
    weight = base_weight * (1 + stagnation_boost)
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized uncertainty across objectives  
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)

        # Scale bonus by acquisition strength: less boost to already strong candidates
        acq_value = cand["acq_value_norm"]
        intensity_factor = 1.0 + (2.0 * (1 - acq_value)) 
        
        values.append(weight * sigma_norm * intensity_factor)
    return values