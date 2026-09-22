def score_pool(context):
    """Blend acquisition value with adaptive uncertainty and inverse coverage gap to dynamically balance exploration vs exploitation based on front sparsity."""
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
        
    scores = []
        
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand['gp_posterior'] 
        
        mu_sum = sum(gp_posterior[name]["mean"] for name in names)
                
        # Uncertainty bonus scaled by campaign phase and front sparsity
        sigma_normed = np.mean([gp_posterior[name]["std"]/context["pareto_front_range"][name] 
                               for name in names])
        
        # Use the inverse of coverage gap (how far from existing points) to modulate uncertainty weight  
        if novel_scores[i] > 0:
            front_sparsity_factor = 1. / novel_scores[i]
        else: 
            front_sparsity_factor = 1.
            
        # Combine progress-aware exploration with sparsity-driven exploitation
        exploit_weight = min(1., max(0., (2 * progress) - 1))  
        
        uncertainty_bonus = sigma_normed * (exploit_weight + 0.5*(front_sparsity_factor-1))
            
        combined_score = acq_scores[i] + uncertainty_bonus
        
        scores.append(combined_score)
        
    return scores