def score_pool(context):
    """Integrate acquisition value with a dynamic uncertainty-aware diversity bonus and progress-driven exploitation balance."""
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

        # Uncertainty bonus scaled by campaign phase: early explore more,
        # late exploit less uncertain candidates with higher acquisition value 
        sigma_sum_normed = sum(gp_posterior[name]["std"]/context["pareto_front_range"][name] 
                               for name in names)
        
        uncertainty_bonus = 0.5 * (sigma_sum_normed / len(names))
            
        # Diversity bonus: prefer points that are novel and have high acquisition
        diversity_boost = 2.0 * np.exp(-novel_scores[i]) * acq_scores[i]
                
        combined_score = acq_scores[i] + uncertainty_bonus - 0.5 * novelty_distances[i] + diversity_boost
        
        scores.append(combined_score)
        
    return scores