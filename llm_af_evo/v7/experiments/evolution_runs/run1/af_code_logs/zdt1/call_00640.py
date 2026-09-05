def modifier(context):
    """Add correction based on how much each candidate's predicted objectives would shift the hypervolume expansion frontier."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        pred_obj = [gp[name]["mean"] for name in names]
        
        # Estimate how much hypervolume would be gained if this candidate were observed
        # by computing the difference between reference point and current pareto front,
        # then using that to compute a relative expansion factor from cand's prediction.
        hv_before = 1.0  
        hv_after = 1.0
        
        for i, name in enumerate(names):
            if pred_obj[i] > ref_point[i]: 
                continue
            new_ref_i = max(ref_point[i], pred_obj[i])
            
            # Simplified hypervolume calculation: assume uniform distribution of front points,
            # and compute the change from using a better reference point for this objective.
            hv_before *= (ref_point[i] - min([p[i] if p[i] < ref_point[i] else ref_point[i]
                                              for p in context["pareto_front"]])) 
                
        values.append(hv_after / max(1e-8, hv_before) * cand["acq_value_norm"])
        
    return [val - np.mean(values) for val in values]  # Normalize to avoid overfitting the scale