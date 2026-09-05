def score_pool(context):
    """Blend acquisition value with inverse distance to nearest observed point and adaptive uncertainty weighting."""
    names = context["objective_names"]
    
    # Use normalized acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute squared distances from each candidate's features to the closest observation
    X_obs = context["X_obs"]
    if len(X_obs) == 0:
        novelty_scores = np.zeros(len(context["pool"]))
    else:
        candidates_x = np.array([cand['x'] for cand in context["pool"]])
        # Vectorized distance computation: each candidate vs all observations
        diff_sq = (candidates_x[:, None, :] - X_obs[None, :, :]) ** 2 
        distances_squared = np.sum(diff_sq, axis=2)  
        min_distances_sq = np.min(distances_squared, axis=1)
        
        # Invert to get novelty: smaller distance -> lower score
        novel_scores_raw = 1.0 / (min_distances_sq + 1e-8) 
        # Normalize between [0., 1.] for blending  
        if len(novel_scores_raw) > 1:
            max_nov = np.max(novel_scores_raw)
            min_nov = np.min(novel_scores_raw)
            
            if abs(max_nov - min_nov) < 1e-8: 
                novelty_scores = np.ones_like(novel_scores_raw)
            else:
                novelty_scores = (novel_scores_raw - min_nov)/(max_nov-min_nov)
        else:
             novelty_scores = novel_scores_raw
    
    # Compute uncertainty scores using a decaying function of progress
    campaign_progress = context["campaign"]["progress"]
    
    if 0.5 <= campaign_progress < 1: 
         ucb_weight = (2 - 2 * campaign_progress)  
     elif campaign_progress >= 1:
          ucb_weight = 0.
     else:
          # Early stages -> more uncertainty weighting
           ucb_weight = max(0., min(campaign_progress, 1.))  
    
    scores = []
    
    for i in range(len(context["pool"])):
        cand = context["pool"][i]
        
        gp_posterior = cand['gp_posterior']
 
         # Sum of normalized GP stds across objectives  
        total_std_normed = sum(gp_posterior[name]["std"]/context["pareto_front_range"][name] for name in names)
         
          final_score = acq_scores[i] + \
                       0.4 * novelty_scores[i]* (1 - ucb_weight) +\
                        ucb_weight*total_std_normed
        
        scores.append(final_score)

    return scores