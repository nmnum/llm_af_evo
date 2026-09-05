def score_pool(context):
    """Blend acquisition value with a progress-adaptive uncertainty term and inverse novelty penalty to encourage diverse exploration early, then focused exploitation later."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute distances from all previously observed points
    X_obs = context["X_obs"]
    if len(X_obs) > 0:
        novelty_distances = []
        for i, cand in enumerate(context["pool"]):
            x_cand = np.array(cand['x'])
            dists_to_observed = [np.linalg.norm(x_cand - x_obs) for x_obs in X_obs]
            min_dist = min(dists_to_observed)
            novelty_distances.append(min_dist)

        # Normalize distances to [0, 1] scale
        if max(novelty_distances) > 0:
            novel_scores = np.array(novelty_distances) / max(novelty_distances)
        else:
            novel_scores = np.zeros_like(novelty_distances)
    else: 
        novel_scores = np.zeros(len(context["pool"]))

    # Progress-aware scaling of uncertainty bonus
    progress = context['campaign']['progress']
    
    # Early in campaign, boost exploration; later focus on exploitation  
    exploit_weight = 1.0 - min(1.0, max(0., (2 * progress) - 1)) 
    explore_weight = 1.0 - exploit_weight
    
    scores = []
        
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand['gp_posterior']
                
        # Uncertainty bonus scaled by campaign phase
        sigma_sum_normed = sum(gp_posterior[name]["std"]/context["pareto_front_range"][name] 
                               for name in names)
        
        uncertainty_bonus = explore_weight * (sigma_sum_normed / len(names))
            
        # Inverse novelty penalty — penalize candidates too close to previous observations  
        novel_penalty = 0.25 * novel_scores[i]
                
        combined_score = acq_scores[i] + uncertainty_bonus - novel_penalty
        
        scores.append(combined_score)
        
    return scores