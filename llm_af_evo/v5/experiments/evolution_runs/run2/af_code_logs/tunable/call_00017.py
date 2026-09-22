def score_pool(context):
    """Blend acquisition value with uncertainty and novelty-aware scoring to improve exploration-exploitation balance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Normalize acq_value_norm to [0, 1] across the pool (already done)
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Add uncertainty component: sum of normalized stds
    uncert_scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names)
        uncert_scores.append(sigma_sum)

    # Normalize novelty scores (distance to nearest observed point)  
    X_obs = context['X_obs']
    novelties = []
    
    if len(X_obs) > 0:
        for cand in context["pool"]:
            x_cand = np.array(cand["x"])
            dists = np.linalg.norm(x_cand - X_obs, axis=1)
            min_dist = np.min(dists)
            # Normalize by max observed distance across features
            if len(X_obs) > 0:
                obs_range = np.max(X_obs, axis=0) - np.min(X_obs, axis=0)
                norm_factor = np.linalg.norm(obs_range[obs_range>0])
                min_dist /= (norm_factor + 1e-8)
            
            novelties.append(min_dist)

        # Normalize novelty scores to [0,1] range
        max_novelty = max(novelties) if any(x > 0 for x in novelties) else 1.0  
        normalized_novelties = np.array([n / (max_novelty + 1e-8) for n in novelties])
    else:
        # No observations yet, assign all same novelty score
        normalized_novelties = [0.] * len(context["pool"])

    
    progress_factor = context['campaign']['progress']
  
    exploitation_weight = max(0.5 + 2*progress_factor - 1., 0.)
    exploration_weight = min(max((1.-progress_factor) / (expansion_rate+1e-8), 0.), 1.) if 'expansion_rate' in locals() else 0.
    
    # Final blend
    final_scores = []
    for i, cand in enumerate(context["pool"]):
        base_score = float(base_scores[i])
        
        ucb_component = (2. * uncert_scores[i]) / len(names)
        novelty_component = normalized_novelties[i]
            
        score = exploitation_weight*base_score + exploration_weight*(ucb_component)  #+ 0.5*novelty_component
        
        final_scores.append(score)

    return final_scores