def modifier(context):
    """Adaptive uncertainty bonus scaled by how close candidate is to expanding the dominated hypervolume region."""
    names = context["objective_names"]
    pf = context["pareto_front"] 
    ref_point = context["ref_point"]

    values = []
    
    # Compute front range for normalization
    if len(pf) == 0:
        front_ranges = [1.0] * len(names)
    else:  
        front_min = np.min(pf, axis=0)
        front_max = np.max(pf, axis=0)
        front_ranges = (front_max - front_min + 1e-8).tolist()
    
    progress_factor = 1.0 - context["campaign"]["progress"]
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        mean_vec = np.array([gp[name]["mean"] for name in names])
        
        # Estimate hypervolume expansion potential
        hv_potential = 0.0
        
        if len(pf) > 0:
            # Check dominance relative to current front  
            is_dominated = False
            
            dominates_some = any(
                all(mean_vec[i] >= pf_point[i] - 1e-8 for i in range(len(names))) 
                and not all(mean_vec[i] == pf_point[i] for i in range(len(names)))
                for pf_point in pf
            )
            
            hv_potential += float(not dominates_some)
        else:
            # No front yet, any candidate adds value  
            hv_potential = 1.0

        sigma_sum = sum(gp[name]["std"] for name in names) 
                
        if len(pf) > 0:    
            norm_pos = (mean_vec - front_min) / (front_max - front_min + 1e-8)
            
            # Coverage score based on distance from center
            coverage_score = np.prod(1.0 - abs(norm_pos - 0.5))
                
        else:
            coverage_score = 1.0
            
        
        bonus_factor = max(coverage_score, 0.2) 
          
        values.append(sigma_sum * hv_potential * bonus_factor * progress_factor)

    return values