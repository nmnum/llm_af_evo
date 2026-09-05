def modifier(context):
    """Add a front-expansion bonus scaled by how much each candidate's prediction would extend beyond current Pareto front boundaries."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    ref_point = context["ref_point"]
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        # Get predicted means
        pred_means = np.array([gp[name]["mean"] for name in names])
        
        # Compute hypervolume expansion if this candidate were observed
        hv_improvement = 0.0
        
        # For each objective, check how much farther the prediction extends beyond front  
        for i, obj_name in enumerate(names):
            pred_val = pred_means[i]
            
            # If predicted value is better than current Pareto frontier's boundary (in that dimension)
            if pf.shape[0] > 0:
                max_front_val = np.max(pf[:,i])
                
                # Bonus only applies when prediction extends beyond front
                if pred_val > max_front_val and pred_val <= ref_point[i]:
                    bonus_distance = min(pred_val - max_front_val, 
                                        (ref_point[i]-max_front_val) * 0.5)
                    
                    hv_improvement += bonus_distance / (ref_point[i] - max_front_val)

        # Normalize by number of objectives
        values.append(hv_improvement / len(names))
    
    return values