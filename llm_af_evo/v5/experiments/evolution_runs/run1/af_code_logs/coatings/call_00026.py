def score_pool(context):
    """Score candidates by acquisition value adjusted for novelty and uncertainty, with dynamic weighting based on progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign = context["campaign"]
    
    # Dynamic weight: start with more exploitation (acq) early, shift to exploration (novelty + uncertainty)
    p = campaign["progress"] 
    w_acq = 0.7 * (1 - p) + 0.3
    w_novelty = 0.2 * p  
    w_uncertainty = 0.1 * p
    
    scores = []
    
    # Compute novelty term: inverse distance to nearest already observed point 
    X_obs = context["X_obs"]
    if len(X_obs) == 0:
        novel_scores = [0.] * len(context["pool"])
    else:
        distances_to_observed = np.array([
            min(np.linalg.norm(cand['x'] - x_obs) for x_obs in X_obs)
                for cand in context["pool"]])
        
        # Normalize novelty score to be between 0 and 1
        max_dist = np.max(distances_to_observed)
        if max_dist > 0:
            novel_scores = (max_dist - distances_to_observed + 1e-8) / (max_dist + 1e-8)
        else: 
            novel_scores = [0.] * len(context["pool"])
    
    for i, cand in enumerate(context["pool"]):
        acq_val_norm = cand["acq_value_norm"]
        
        # Uncertainty bonus
        sigma_bonus = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                          for name in names)
          
        score = w_acq * acq_val_norm + \
                w_novelty * novel_scores[i] + \
                w_uncertainty * sigma_bonus
        
        scores.append(score)

    return scores