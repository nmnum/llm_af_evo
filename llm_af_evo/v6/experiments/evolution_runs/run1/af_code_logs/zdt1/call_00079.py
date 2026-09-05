def modifier(context):
    """Add a front-coverage gap bonus scaled by acquisition strength and stagnation level."""
    import numpy as np
    
    pool = context["pool"]
    pareto_front = context["pareto_front"] 
    names = context['objective_names']
    
    stagnant_batches = context["campaign"]["stagnant_batches"]  
    base_weight = 0.25
    scaling_factor = min(stagnant_batches / 3.0, 1.0)
    weight = base_weight * scaling_factor
    
    values = []
    
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Estimate how far this candidate's predicted objectives are from the Pareto front
        pred_obj = [gp_posterior[name]["mean"] for name in names]
        
        if len(pareto_front) == 0: 
            gap_distance = np.linalg.norm(pred_obj)
        else:
            distances_to_pf = []
            
            # Compute distance to each point on pareto front (using L2 norm)
            for pf_point in pareto_front:
                dist = np.sqrt(sum((pred_obj[i] - pf_point[i])**2 for i in range(len(names))))
                distances_to_pf.append(dist)

            gap_distance = min(distances_to_pf) if len(distances_to_pf) > 0 else float('inf')
            
        # Normalize by the observed front spread (to make it dimensionless)
        front_range = context["pareto_front_range"]
        
        normalized_gap = sum(gap_distance / front_range[name] for name in names)

        acq_value_norm = cand['acq_value_norm']
 
        values.append(weight * normalized_gap * acq_value_norm) 
        
    return values