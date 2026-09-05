def score_pool(context):
    """Blend acquisition value with a dynamic uncertainty term and Pareto front density-aware novelty to balance exploration and exploitation."""
    
    names = context["objective_names"]
    pf = context["pareto_front"] 
    ref_point = np.array([context['ref_point_by_name'][name] for name in names])
    acq_values = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Compute novelty scores based on x-space distances to observed points
    X_obs = context["X_obs"]
    if len(X_obs) > 0:
        novel_scores = []
        for i, cand in enumerate(context["pool"]):
            x_cand = np.array(cand['x'])
            dists_to_observed = [np.linalg.norm(x_cand - x_obs) for x_obs in X_obs]
            min_dist = min(dists_to_observed)
            novel_scores.append(min_dist)

        # Normalize novelty scores to [0, 1] 
        max_novelty = max(novel_scores)
        if max_novelty > 0:
            normalized NovelScores = np.array(novel_scores) / max_novelty
        else:
            normalized_NovelScores = np.zeros_like(novel_scores)
    else: 
        normalized_NovelScores = np.zeros(len(context["pool"]))
    
    # Compute uncertainty bonus with progress-aware scaling  
    campaign_progress = context['campaign']['progress']
        
    # Early on, emphasize exploration; later favor exploitation
    explore_weight = max(0., 1. - 2 * campaign_progress) 
   
    scores = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand['gp_posterior']

        sigma_sum_normed = sum(gp_posterior[name]["std"]/context["pareto_front_range"][name] 
                               for name in names)
        
        uncertainty_bonus = explore_weight * (sigma_sum_normed / len(names))
                
        # Combine acquisition value with scaled uncertainty bonus and inverse novelty
        final_score = acq_values[i] + 0.5 * uncertainty_bonus - \
                      normalized_NovelScores[i]
                      
        scores.append(final_score)

    return scores