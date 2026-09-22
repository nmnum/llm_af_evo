def modifier(context):
    """Add a correction term that rewards candidates with objective predictions lying far outside the current Pareto front's extent in any dimension."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    ref_point = context["ref_point"]
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        pred_means = np.array([gp[name]["mean"] for name in names])
        
        # Compute how much each objective's predicted mean exceeds the front bounds
        extent_excess = 0.0  
        for i, name in enumerate(names):
            if len(pf) > 0:
                min_front_val = pf[:,i].min()
                excess = max(0., pred_means[i] - min_front_val)
            else: 
                # If no front yet, use reference point as a baseline
                excess = max(0., pred_means[i] - ref_point[i])
            
            extent_excess += excess
        
        values.append(extent_excess * 0.1) 
    
    return values