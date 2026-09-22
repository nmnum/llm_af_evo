def modifier(context):
    """Adaptive hypervolume gap bonus: reward candidates that would expand the dominated region more than current front's reach."""
    import numpy as np
    
    pool = context["pool"]
    pareto_front = context["pareto_front"] 
    ref_point = context["ref_point"]
    
    # Normalize reference point for each objective
    names = context['objective_names']
    front_range = {name: context["pareto_front_range"][name] for name in names}
        
    values = []
    for cand in pool:
        gp_posterior = cand["gp_posterior"] 
        
        # Estimate candidate's predicted objectives (use mean)
        pred_obj = [gp_posterior[name]["mean"] for name in names]
    
        # Compute hypervolume contribution of this point if added to front
        hv_contribution = 0.0
        
        # If we have a pareto_front, compute the gap from current best 
        if len(pareto_front) > 1:
            cand_pred = np.array(pred_obj)
            
            for i in range(len(ref_point)):
                ref_val = ref_point[i]
                
                # Compute distance to front's boundary (min of all points' values on this axis, or reference point).
                min_on_axis = float('inf')
                if len(pareto_front) > 0:
                    vals_in_PF = pareto_front[:, i] 
                    min_on_axis = np.min(vals_in_PF)
                    
                # The "gap" is how much more dominated space we can cover from this candidate's prediction
                gap_to_ref = ref_val - cand_pred[i]
                
                if (min_on_axis < 1e-6) or abs(min_on_axis) <= min(0.5 * front_range[names[i]], 
                    np.abs(ref_point[i] / 2)):
                    # If it is too close to the reference, don't add a large contribution.
                    gap_to_ref = max(gap_to_ref - (ref_val*1e-3), 0)
                else:
                    pass
                
                hv_contribution += abs(max(0.0, gap_to_ref))
                
        elif len(pareto_front) == 1 and not np.allclose(pred_obj, pareto_front[0]):
            # Only one point in front: compare candidate's objective with reference
            cand_pred = np.array(pred_obj)
            
            for i in range(len(ref_point)):
                gap_to_ref = ref_point[i] - max(0.0,cand_pred[i])
                
                hv_contribution += abs(max(gap_to_ref, 0))
        else:
             # Empty front or identical point: no contribution
              pass
            
        
         # Compute uncertainty bonus based on std of objectives (normalized)
        sigma_norm = sum(gp_posterior[name]["std"] for name in names) 
        
        if hv_contribution > np.mean(ref_point):
            weight_hv_gap  = min(0.5, hv_contribution / max(np.max(pareto_front),1e-6)) * sigma_norm
        else:
             # Small HV contribution: only small bonus 
              weight_hv_gap = (hv_contribution/max(np.sum(front_range.values()), 1.e-4) ) **2   *sigma_norm

        
         values.append(weight_hv_gap)
         
    return np.array(values).tolist()