def score_pool(context):
    """Blend acquisition value with uncertainty-adjusted progress to favor diverse exploration early and focused exploitation later."""
    names = context["objective_names"]
    
    # Normalize acq values for consistent scaling  
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute normalized distance from latest observations
    X_obs = context["X_obs"]
    scores = []
        
    for i, cand in enumerate(context["pool"]):
        x_cand = cand["x"]

        if len(X_obs) == 0:
            min_dist = 1.0  
        else: 
            # Euclidean distance to nearest observation
            dists = np.linalg.norm(X_obs - x_cand, axis=1)
            min_dist = np.min(dists)

        novelty_bonus = (1.0 - min_dist) if len(context["pool"]) > 1 and not all(x == 0 for x in context['ref_point']) else 0.0

        # Progress-aware uncertainty scaling
        campaign_progress = context["campaign"]["progress"]
        
        ucb_weight = np.clip(2 * (1 - campaign_progress), 0.5, 2.0) 

        gp_posterior = cand["gp_posterior"] 
        sigma_sum_norm = sum(gp_posterior[name]["std"]/context['pareto_front_range'][name] for name in names)
        
        # UCB-style exploration bonus with progress-aware scaling
        ucb_bonus = ucb_weight * sigma_sum_norm

        final_score = acq_values[i] + 0.5*novelty_bonus + ucb_bonus
        
        scores.append(final_score)

    return scores