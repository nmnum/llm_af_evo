def modifier(context):
    """Adaptive front-expansion bonus scaled by uncertainty and acquisition strength."""
    
    import numpy as np
    
    names = context["objective_names"]
    if len(context.get("X_obs", [])) == 0:
        return [0.] * len(context["pool"])
        
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    front_points = context["pareto_front"] 
    progress = context["campaign"]["progress"]
    
    values = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        means = np.array([gp_posterior[name]["mean"] for name in names])

        # Estimate how much the candidate's mean would expand front coverage
        hv_impact = 0.0
        
        dominated_by_front = False  
        for pf_point in front_points:
            if all(pf_point >= means):
                dominated_by_front = True
                
        if not dominated_by_front: 
             ref_dists_to_mean = np.maximum(ref_point - means , 0) 
             hv_impact += max(np.prod(ref_dists_to_mean),1e-8)
        
        # Scale by uncertainty and acquisition value
        sigma_sum = sum(gp_posterior[name]["std"] for name in names)
        acq_value_norm = cand["acq_value_norm"]
        
        weight_factor = (0.5 + 0.5 * progress) / (sigma_sum + 1e-6)
        bonus = hv_impact * weight_factor
        
        values.append(bonus)

    max_val = float(np.max(values))
    
    if abs(max_val) < 1e-8:  
       return [val / (max_val + 1e-6 ) for val in values]
        
    normalized_values = []
    for v in values:
        norm_v = float(v)/ max_val 
        normalized_values.append( norm_v )
    
    # Normalize to prevent over-scaling
    final_vals = np.array(normalized_values)
    return (final_vals / 2.).tolist()