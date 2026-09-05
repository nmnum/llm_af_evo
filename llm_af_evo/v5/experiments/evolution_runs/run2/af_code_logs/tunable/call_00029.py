def score_pool(context):
    """Blend acquisition value with uncertainty and novelty-aware scoring to improve exploration-exploitation balance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    X_obs = context['X_obs']
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    
    # Uncertainty bonus (UCB-style) weighted by progress
    uncert_bonus = 0.5 
    unc_scores = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        unc_scores.append(uncert_bonus * sigma_sum)

    # Novelty bonus based on distance to nearest observed point
    nov_penalty = 0.2 
    if len(X_obs) > 0:
        obs_dists = []
        for cand in context["pool"]:
            x_cand = np.array(cand['x'])
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)
            # Normalize by feature dimensionality (assumed [0, 1] range)  
            obs_dists.append(min_dist / len(x_cand))
        nov_scores = [-nov_penalty * d for d in obs_dists]
    else:
        nov_scores = [0.0] * len(context["pool"])
        
    # Progress-aware blending
    progress = context['campaign']['progress']
    
    exploitation_weight = max(0.3 + 1.2*progress, 0.) 
    exploration_weight = min(max((1.-progress) / (0.5+1e-8), 0.), 1.)
        
    # Final score combining all components
    final_scores = []
    for i in range(len(context["pool"])):
        exploitation_component = exploitation_weight * acq_scores[i]
        exploration_component = exploration_weight * unc_scores[i] 
        novelty_component = nov_scores[i]

        total_score = exploitation_component + exploration_component + novelty_component
        
        final_scores.append(total_score)
        
    return final_scores