def modifier(context):
    """Adaptive uncertainty bonus with front coverage gap adjustment and acquisition-strength scaling."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Dynamic base weight that decreases as campaign progresses
    progress = context["campaign"]["progress"]
    base_weight = 0.3 * (1 - progress)
    
    # Stagnation scaling factor: more boost when stalled  
    stagnant_batches = context["campaign"]["stagnant_batches"]
    scaling_factor = min(stagnant_batches / 5.0, 1.0) 
    
    weight = base_weight * scaling_factor
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Scale bonus by acquisition strength (less boost to high-acq candidates)  
        acq_value = cand["acq_value_norm"]
        adjustment_factor = 1.0 + 2.0 * (1 - acq_value)
        
        # Combine uncertainty with front coverage gap
        if len(context["Y_obs"]) > 0:
            y_min = np.array([context["ref_point_by_name"][name] for name in names])
            y_max = np.array([max(context["Y_obs"][:, i]) for i, name in enumerate(names)])
            ranges = y_max - y_min
            normalized_y_obs = (context["Y_obs"] - y_min) / ranges
            
            mean_vec = np.array([gp[name]["mean"] for name in names])
            norm_mean = (mean_vec - y_min) / ranges 
            distances = np.linalg.norm(normalized_y_obs - norm_mean, axis=1)
            min_distance = np.min(distances)
            
            # Front coverage gap bonus: more reward to candidates near uncovered regions
            front_coverage_bonus = 0.5 * (1 - min_distance if min_distance < 1 else 0) 
        else:
            front_coverage_bonus = 0
            
        values.append(weight * sigma_norm * adjustment_factor + front_coverage_bonus)
    return values