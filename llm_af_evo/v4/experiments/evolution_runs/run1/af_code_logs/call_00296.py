def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with an adaptive uncertainty-adjusted novelty bonus that scales based on campaign progress and stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    
    # Determine exploration weight: higher early, lower late
    prog = context["campaign"]["progress"] 
    stagnant = context["campaign"]["stagnant_batches"]

    if len(context['Y_obs']) < 2:
        w_explore = 1.0  
    else:
        base_weight = max(0., 1 - prog * (0.5 + min(stagnant/3, 0.7)))
        # Increase weight slightly when stagnant
        explore_factor = 1.0 if stagnate == 0 else 2.0 / (stagnant + 1)
        w_explore = base_weight * explore_factor
    
    for cand in context["pool"]:
        acq_norm = cand['acq_value_norm']
        
        # Compute uncertainty-adjusted novelty
        gp_posterior = cand["gp_posterior"]
        
        mu_sum = sum(gp_posterior[name]["mean"]  for name in names)
        sigma_scaled = sum(gp_posterior[name]["std"] / front_range[name]   for name in names) 
        
        if len(context['Y_obs']) == 0:
            novelty_score = 1.0
        else: 
            cand_x = cand["x"]
            
            # Euclidean distance to nearest observed point (in feature space)
            distances_sq = np.sum((context["X_obs"] - cand_x) ** 2, axis=1)
            min_dist_sq = np.min(distances_sq)
        
            novelty_score = max(0., 1.0 / (min_dist_sq + 1e-9))
            
        # Final score: blend of acquisition value and uncertainty-novelty
        ucb_term = w_explore * sigma_scaled 
        novel_term = (1 - w_explore) * np.log(n novelty_score + 1)
        
        scores.append(acq_norm + ucb_term + novel_term)

    return scores