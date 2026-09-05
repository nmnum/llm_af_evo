def modifier(context):
    """Adaptive hypervolume expansion bonus scaled by prediction confidence and stagnation level."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"] 
    pareto_front = context["pareto_front"]
    
    # Compute reference point normalized by front range
    ranges = [context["pareto_front_range"][name] for name in names]
    norm_ref = ref_point / ranges
    
    stagnant_batches = context["campaign"]["stagnant_batches"]
    base_weight = 0.3 * min(stagnant_batches / 5.0, 1.0)
    
    values = []
    for cand in context['pool']:
        gp_posterior = cand["gp_posterior"]

        # Estimate mean prediction vector
        pred_vec = np.array([gp_posterior[name]["mean"] for name in names])
        
        # Normalize candidate's predicted point  
        norm_pred = pred_vec / ranges
        
        # Compute hypervolume contribution if this were the only new point added to front (simplified)
        hv_contribution = 1.0
        for i, _ in enumerate(names):
            diff_to_ref = max(0., norm_ref[i] - norm_pred[i])
            hv_contribution *= diff_to_ref
            
        # Scale bonus by how much candidate's prediction is uncertain  
        sigma_norm = sum(gp_posterior[name]["std"] / ranges[i] 
                         for i, name in enumerate(names))
        
        values.append(base_weight * (sigma_norm + 0.5) * hv_contribution)
    
    return values