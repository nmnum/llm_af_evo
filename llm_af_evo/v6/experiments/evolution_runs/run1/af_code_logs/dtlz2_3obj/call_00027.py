def modifier(context):
    """Add an adaptive exploration bonus based on the gap between predicted objectives and current Pareto front, with decay over stagnation."""
    import numpy as np
    
    pool = context["pool"]
    pareto_front = context["pareto_front"] 
    names = context['objective_names']
    ref_point = context["ref_point"]
    
    stagnant_batches = context["campaign"]["stagnant_batches"]  
    weight_factor = 0.4 * max(0, 1 - stagnant_batches / 8.0)
    
    values = []
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Compute the gap from Pareto front
        predicted_obj = np.array([gp_posterior[name]["mean"] for name in names])
        min_gap_to_front = float('inf')
        
        if len(pareto_front) > 0: 
            for pf_point in pareto_front:
                gap = sum(max(0, (pf_point[i] - predicted_obj[i]) / ref_point[i])  
                          for i in range(len(names)))
                min_gap_to_front = min(min_gap_to_front, gap)
        
        # If no front exists or candidate is far from it
        if np.isinf(min_gap_to_front):
            bonus = 1.0 
        else:
            bonus = max(0., (min_gap_to_front - 0.2)) / 5.
            
        values.append(weight_factor * bonus)
        
    return values