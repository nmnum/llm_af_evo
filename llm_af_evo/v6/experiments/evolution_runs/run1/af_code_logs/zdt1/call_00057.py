def modifier(context):
    """Add a dynamic uncertainty bonus that increases with stagnation and decreases with acquisition strength, encouraging exploration of uncertain candidates only when progress has stalled."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base uncertainty term for each candidate (normalized sum of std)
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)  
        values.append(sigma_norm)

    # Normalize uncertainties to [0, 1]
    if max(values) > min(values):
        u_values = [(v - min(values)) / (max(values) - min(values)) for v in values]
    else:
        u_values = [0.0] * len(values)
        
    # Scale by stagnation factor: higher when campaign is stagnant
    s_factor = 1 + context["campaign"]["stagnant_batches"] ** 0.5
    
    # Final correction scales uncertainty bonus with stagnation but caps it per candidate  
    final_values = []
    for i, u in enumerate(u_values):
        acq_norm = context["pool"][i]["acq_value_norm"]
        
        # Only apply boost to candidates that aren't already very strong
        strength_factor = 1 - min(acq_norm * 2.0, 1.0)  
      
        bonus = s_factor * u * strength_factor
        
        final_values.append(bonus)
    
    return final_values