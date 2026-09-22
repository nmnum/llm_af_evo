def modifier(context):
    """Front expansion gap bonus scaled by stagnation and acquisition strength."""
    import numpy as np
    
    pool = context["pool"]
    pareto_front = context["pareto_front"] 
    ref_point = context["ref_point"]
    names = context['objective_names']
    
    stagnant_batches = context["campaign"]["stagnant_batches"]  
    base_weight = 0.3
    scaling_factor = min(stagnant_batches / 5.0, 1.0)
    weight = base_weight * scaling_factor
    
    values = []
    
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Estimate how much the candidate would expand hypervolume
        pred_means = np.array([gp_posterior[name]["mean"] for name in names])
        gaps = ref_point - pred_means
        
        # Only consider expansion potential if we're not already at reference point  
        gap_bonus = 0.0
        if any(gap > 1e-6 for gap in gaps):
            front_expansion_gap = np.min(np.maximum(0, gaps))
            
            # Scale bonus by how strong the acquisition value is and stagnation level 
            acq_value_norm = cand["acq_value_norm"]
            gap_bonus = weight * (front_expansion_gap / max(gaps)) * acq_value_norm
            
        values.append(float(gap_bonus))

    return values