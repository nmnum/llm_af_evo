def modifier(context):
    """Add an adaptive uncertainty bonus that rewards exploration towards under-covered regions of the objective space."""
    names = context["objective_names"]
    pf = context["pareto_front"] 
    ref_point = context["ref_point"]
    
    # Compute coverage statistics across all objectives
    if len(pf) == 0:
        front_ranges = [1.0] * len(names)
    else:  
        front_min = np.min(pf, axis=0)
        front_max = np.max(pf, axis=0)
        front_ranges = (front_max - front_min + 1e-8).tolist()
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Estimate candidate's objective vector
        mean_vec = np.array([gp[name]["mean"] for name in names])
            
        # Compute how much this point would extend the dominated region 
        hv_expansion_potential = 0.0
        
        if len(pf) > 0:
            # Check dominance relative to current front  
            is_dominated = False
            for pf_point in pf:   
                dominates = all(mean_vec[i] >= pf_point[i] - 1e-8 for i in range(len(names)))
                if dominates and not all(mean_vec[i] == pf_point[i] for i in range(len(names))):
                    # Strictly improves over some point
                    is_dominated = True 
                    break
                    
            hv_expansion_potential += float(not is_dominated)
            
        else:
            # No front yet - any candidate adds value  
            hv_expansion_potential = 1.0
            
        
        # Bonus based on uncertainty and coverage position   
        sigma_sum = sum(gp[name]["std"] for name in names) 
                
        # Scale bonus by how under-covered this region is (based only on current objectives)
        if len(pf) > 0:
            norm_pos = np.array(mean_vec - front_min) / (front_max - front_min + 1e-8)
            
            coverage_score = np.prod(1.0 - np.abs(norm_pos - 0.5))  
                
            # Reward candidates in under-covered corners
            bonus_factor = max(coverage_score, 0.2) 
        else:
            bonus_factor = 1.0
            
        
        values.append(sigma_sum * hv_expansion_potential * bonus_factor * 0.3)
    
    return values