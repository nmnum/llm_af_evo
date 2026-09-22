def score_pool(context):
    """Phase-aware blend of acquisition value and uncertainty with novelty boost during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Decay factor for uncertainty term based on campaign progress
    ucb_weight = 1.0 - progress
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        acq_norm = cand["acq_value_norm"]
        
        # Uncertainty normalized by front range  
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Novelty bonus if stagnant
        novelty_bonus = 0.0
        if stagnant_batches >= 3:
            min_dist_to_observed = np.min(np.linalg.norm(cand["x"] - context["X_obs"], axis=1))
            novelty_bonus = (min_dist_to_observed / np.sqrt(len(context["objective_names"]) * 6)) ** 2
        
        # Combine with phase-aware UCB weight and stagnation bonus
        score = acq_norm + ucb_weight * sigma_norm + 0.5 * novelty_bonus 
        scores.append(score)
        
    return scores