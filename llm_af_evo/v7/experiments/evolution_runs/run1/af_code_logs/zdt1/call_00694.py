def modifier(context):
    """Add bonus for candidates that are diverse in objective space relative to current Pareto front, penalizing those aligned with existing improvement directions."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    ref_point = context["ref_point"]
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        # Estimate candidate's objective vector
        mean_vec = np.array([gp[name]["mean"] for name in names])
        
        # Compute distance to Pareto front (hypervolume gap)
        if len(pf) == 0:
            hv_gap = 1.0  
        else:   
            # Use reference point as dominance baseline 
            dominated_by_front = False
            for pf_point in pf:
                if all(mean_vec[i] >= pf_point[i] - 1e-8 for i in range(len(names))):
                    dominated_by_front = True
                    break
            
            hv_gap = float(not dominated_by_front)
            
        # Compute direction from reference to candidate's mean  
        ref_to_mean = mean_vec - ref_point
        
        # If close to front, check if this point extends the improvement frontier 
        bonus = 0.0        
        for pf_point in pf:
            diff = (mean_vec - pf_point) / (ref_point - pf_point + 1e-8)
            
            # Check how much candidate improves over existing points
            improve_ratio = np.mean(diff[diff > 0]) if len(np.where(diff>0)[0]) else 0
            
            bonus += max(0, improve_ratio * hv_gap) 
        
        values.append(bonus / (len(pf)+1)) 

    return values