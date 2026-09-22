def modifier(context):
    """Adaptively blend uncertainty bonus with novelty reward based on stagnation level."""
    pool = context["pool"]
    X_obs = context["X_obs"] 
    Y_obs = context["Y_obs"]  
    pareto_front = context["pareto_front"]
    
    names = context['objective_names']
    front_range = context["pareto_front_range"]
    stagnant_batches = context["campaign"]["stagnant_batches"] 
    
    values = []
    
    for cand in pool:
        gp_posterior = cand["gp_posterior"]

        # Uncertainty bonus
        sigma_norm = sum(gp_posterior[name]["std"] / front_range[name] for name in names) 
        
        # Novelty reward: inverse distance to nearest observation  
        cand_x = cand["x"]
        min_dist_sq = float('inf')
        
        if len(X_obs) > 0:
            distances_sq = np.sum((X_obs - cand_x)**2, axis=1)
            min_dist_sq = np.min(distances_sq)

        # Convert distance to a reward (higher when farther from obs)
        novelty_reward = 0.0
        if min_dist_sq != float('inf'):
            dist = np.sqrt(min_dist_sq) 
            novelty_reward = max(0., 1. - dist / 2.)  # Normalize by some reference
            
        scaling_factor = min(stagnant_batches / 3.0, 1.0)
        
        weight_uncertainty = 0.4 * (1 + scaling_factor)
        weight_novelty = 0.6 * (1 + scaling_factor) 
        
        values.append(weight_uncertainty * sigma_norm + weight_novelty * novelty_reward)

    return values