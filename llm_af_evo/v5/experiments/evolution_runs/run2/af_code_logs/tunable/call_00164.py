def score_pool(context):
    """Use acquisition value normalized by predicted objective variance to encourage exploration while maintaining exploitability."""
    names = context["objective_names"]
    
    # Retrieve acq values and GP posteriors 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    gp_posteriors = [cand["gp_posterior"] for cand in context["pool"]]
    
    # Compute variance-weighted acquisition scores
    var_weights = []
    front_range = context["pareto_front_range"]
        
    for i, gp in enumerate(gp_posteriors):
        total_var = sum((gp[name]["std"])**2 / (front_range[name])**2 for name in names)
        if total_var == 0:
            # Avoid division by zero; use a small value
            var_weights.append(1.0)  
        else: 
            inv_total_std = 1.0 / np.sqrt(total_var + 1e-8)   # invert and normalize std deviation to weight acquisition score
            
            # Blend with inverse variance (higher uncertainty gets higher weights)
            weighted_acq = acq_values[i] * max(0, min(inv_total_std, 5)) 
                
        var_weights.append(weighted_acq)

    return var_weights