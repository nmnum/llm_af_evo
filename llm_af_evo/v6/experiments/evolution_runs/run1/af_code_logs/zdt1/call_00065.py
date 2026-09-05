def modifier(context):
    """Penalize candidates with low uncertainty when acquisition value is high and front expansion potential is weak."""
    names = context["objective_names"]
    pareto_front = context["pareto_front"] 
    ref_point = context["ref_point"]
    
    values = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Estimate how much the candidate could expand hypervolume
        hv_expansion_potential = 0.0  
        if len(pareto_front) > 1:
            # Simple proxy: distance from current front to reference point in each objective 
            dist_to_ref = np.array([ref_point[i] - max(pf[i] for pf in pareto_front)
                                   for i in range(len(names))])
            
            hv_expansion_potential = sum(max(0.0, d) * gp_posterior[name]["std"]  
                                        for name, d in zip(names, dist_to_ref))
        
        # Combine acquisition strength with uncertainty and front expansion
        acq_value_norm = cand["acq_value_norm"]
        sigma_sum = sum(gp_posterior[name]["std"] for name in names)
        
        if hv_expansion_potential > 0.1:  
            values.append(0.0) 
        else:
            # Apply penalty when candidate is certain but not expanding front much
            bonus = -max(0., acq_value_norm * sigma_sum / (sigma_sum + 1e-8)) 
            
            if hv_expansion_potential < 0.05 and cand["acq_value_norm"] > 0.7:
                values.append(bonus) 
            else:  
                values.append(-max(0., acq_value_norm * sigma_sum / (sigma_sum + 1e-8)) )
                
    return values