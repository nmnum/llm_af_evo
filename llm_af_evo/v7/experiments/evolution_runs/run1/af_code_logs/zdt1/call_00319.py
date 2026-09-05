def modifier(context):
    """Combine uncertainty bonus with novelty penalization to balance exploration and diversity."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Uncertainty component (UCB-style)
    weight_uncert = 0.3 * max(0.1, 1.0 - context["campaign"]["progress"])
    values = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Base uncertainty bonus
        uncert_bonus = weight_uncert * sigma_norm
        
        # Novelty penalization (similar to parent A but scaled by acquisition value)
        cand_x = cand["x"]
        min_dist_sq = float('inf')
        for prev_cand in context["pool"]:
            if 'selected' not in prev_cand:  # Only consider already selected candidates
                dist_sq = np.sum((cand_x - prev_cand["x"]) ** 2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        
        novelty_penalty = 0.0 
        if min_dist_sq != float('inf'):
            distance = np.sqrt(min_dist_sq)  
            multiplier = (1.0 - np.exp(-distance)) * cand["acq_value_norm"] # Scale by acquisition strength
            novelty_penalty = -(multiplier * 0.2)
        
        values.append(uncert_bonus + novelty_penalty)

    return values