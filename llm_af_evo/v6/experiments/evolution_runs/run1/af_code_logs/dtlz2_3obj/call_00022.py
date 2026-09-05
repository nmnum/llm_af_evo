def modifier(context):
    """Adaptive hypervolume expansion bonus based on predicted objective values and campaign stagnation."""
    import numpy as np
    
    pool = context["pool"]
    pareto_front = context["pareto_front"] 
    ref_point = context["ref_point"]
    
    # Calculate the reference point adjusted for each candidate's mean prediction
    names = context['objective_names']
    front_range = context["pareto_front_range"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    base_weight = 0.3
    
    values = []
        
    for cand in pool:
        gp_posterior = cand["gp_posterior"] 
      
        # Compute predicted hypervolume contribution using reference point
        pred_means = np.array([gp_posterior[name]["mean"] for name in names])
          
        if len(pareto_front) > 0:  
            # Calculate the dominated volume by comparing to current front and ref_point            
            vol_contributions = []
                        
            for i, pf_point in enumerate(pareto_front):
                # Compute hypervolume contribution of this candidate relative to PF point
                lower_bounds = np.minimum(pred_means, pf_point)
                
                if not (pred_means <= pf_point).all():  # Not dominated by front member 
                    vol_contributions.append(np.prod(ref_point - pred_means))
            
            hv_contribution = sum(vol_contributions) / len(pareto_front) if vol_contributions else 0.0
        else:
             # No PF yet, so just use raw hypervolume to reference point  
             hv_contribution = np.prod(ref_point - pred_means)
        
        scaling_factor = min(stagnant_batches / 5.0 + 1e-8, 2.0) 
        weight = base_weight * scaling_factor
        
        values.append(weight *hv_contribution )
            
    return values