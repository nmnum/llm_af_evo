def modifier(context):
    """Adaptive uncertainty bonus with dynamic weighting based on stagnation and acquisition strength."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Dynamic base weight that decreases as campaign progresses
    progress = context["campaign"]["progress"]
    base_weight = 0.3 * (1 - progress)
    
    # Stagnation scaling factor: more boost when stalled
    scaling_factor = min(stagnant_batches / 5.0, 1.0) 
    
    weight = base_weight * scaling_factor
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Scale bonus by acquisition strength (only apply to promising candidates)  
        acq_value = cand["acq_value_norm"]
        adjustment_factor = 1.0 + 2.0 * (1 - acq_value)  # Less boost to high-acq candidates
        
        values.append(weight * sigma_norm * adjustment_factor)
    return values