def modifier(context):
    """Reward candidates with objective means that extend beyond current Pareto front boundaries into unexplored hypervolume regions."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    ref_point = context["ref_point"]
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        pred_means = np.array([gp[name]["mean"] for name in names])
        
        # Compute hypervolume contribution if this candidate were added to the front
        extended_pf = np.vstack([pf, pred_means])  
        hv_improvement = 0.0
        
        try:
            # Simple bounding box approach: how much more volume would be dominated 
            # by a new point beyond current pareto frontier?
            min_vals = np.minimum.reduce(extended_pf)
            
            if len(pf) > 1 and all(min_vals >= ref_point):
                hv_improvement = np.prod(ref_point - min_vals)

        except (ValueError, ZeroDivisionError): 
            pass

        values.append(hv_improvement * 0.25) # Scale to reasonable range
    
    return values