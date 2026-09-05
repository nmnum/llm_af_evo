def modifier(context):
    """Adaptive uncertainty bonus with front-gradient awareness and acquisition-value saturation."""
    import numpy as np
    
    pool = context["pool"]
    pareto_front = context["pareto_front"] 
    names = context['objective_names']
    
    # Compute gradient of the Pareto front in objective space
    if len(pareto_front) < 2:
        return [0.0] * len(pool)
        
    grad_magnitudes = []
    for i, name in enumerate(names):
        pf_vals = pareto_front[:,i]
        grads = np.diff(pf_vals)
        mag = np.mean(np.abs(grads)) if len(grads) > 0 else 0
        grad_magnitudes.append(mag)

    # Use gradient info to scale uncertainty bonus — less exploration near steep gradients  
    front_steepness = sum(grad_magnitudes) / len(names)
    
    values = []
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        sigma_norm = np.mean([gp_posterior[name]["std"] for name in names])
 
        # Scale uncertainty bonus based on front steepness and acquisition value
        base_weight = 0.3 * (1 - front_steepness) 
        
        acq_val = cand["acq_value_norm"]
        
        # Saturation: reduce the effect of this correction if candidate already has high acquisition value  
        saturation_factor = np.exp(-5 * (1 - acq_val)) 
          
        weight = base_weight * sigma_norm * saturation_factor
        
        values.append(weight)
    
    return values