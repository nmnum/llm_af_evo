def modifier(context):
    """Combine uncertainty bonus with novelty penalty to balance exploration and diversity."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Uncertainty-based bonus (UCB-style)
    weight_uncert = 0.3 * max(0, 1 - context["campaign"]["progress"])
    sigma_norms = [sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names) 
                   for cand in context["pool"]]
    
    # Novelty penalty (distance to selected candidates)
    weight_novelty = 0.244
    values = []
    selected_x = []

    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]
        
        # Uncertainty bonus component 
        uncert_bonus = weight_uncert * sigma_norms[i]
        
        # Novelty penalty: compute distance to nearest already-selected candidate
        min_dist_sq = float('inf')
        for prev_x in selected_x:
            dist_sq = np.sum((cand["x"] - prev_x) ** 2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        
        # Apply penalty only to candidates already strong (acq_value_norm > threshold),
        # and scale by how close they are
        acq_val = cand["acq_value_norm"]
        distance = np.sqrt(min_dist_sq) if min_dist_sq != float('inf') else 0.01 
        multiplier_novelty = max(0, (1 - np.exp(-distance)))  
        
        # Only apply novelty penalty to candidates with high acquisition value
        novel_penalty = weight_novelty * multiplier_novelty * (acq_val > 0.5) 
        
        values.append(uncert_bonus - novel_penalty)
    
    return values