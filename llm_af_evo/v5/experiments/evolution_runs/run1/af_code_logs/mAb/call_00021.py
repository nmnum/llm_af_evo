def score_pool(context):
    """Score candidates by acquisition value adjusted for novelty and progress-aware uncertainty."""
    names = context["objective_names"]
    
    # Use acq_value_norm directly as quality signal (already computed via qLogNEHVI)
    qualities = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Normalize acquisition values to [0, 1] range
    q_min, q_max = qualities.min(), qualities.max()
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.full_like(qualities, 0.5)
    else:
        norm_qualities = (qualities - q_min) / (q_max - q_min)

    # Compute novelty scores based on distance to existing observations
    x_observed = context["X_obs"]
    
    if len(x_observed) == 0:
        novel_scores = np.ones(len(context["pool"]))
    else: 
        pool_x = np.array([cand['x'] for cand in context["pool"]])
        
        # Compute pairwise distances between candidates and observed points
        dists_sq = np.sum((pool_x[:, None, :] - x_observed[None, :, :]) ** 2, axis=2)
        min_dists = np.min(dists_sq, axis=1)

        novel_scores = 1.0 / (min_dists + 1e-8) 

    # Normalize novelty scores to [0, 1]
    n_min, n_max = novel_scores.min(), novel_scores.max()
    
    if abs(n_max - n_min) < 1e-9:
        norm_novelty = np.full_like(novelscores, 0.5)
    else:
        norm_novelty = (novel_scores - n_min) / (n_max - n_min)

    # Progress-aware uncertainty adjustment
    campaign_progress = context["campaign"]["progress"]
    
    if campaign_progress < 0.3: 
         # Early stage: favor exploration via higher uncertainty weight  
         u_weight = 1.
    elif campaign_progress > 0.7:
        # Late stage: reduce influence of noise, rely more on acquisition value
        u_weight = 0.2
    else:
        # Middle stages moderate balance between exploitation and exploration 
        u_weight = 0.6

    scores = []
    
    for i in range(len(context["pool"])):
        
        gp_posterior = context['pool'][i]['gp_posterior']
            
        sigma_sum_normed = sum(gp_posterior[name]["std"]
                               / (context["pareto_front_range"][name] + 1e-8)
                                for name in names)

        # Mix acquisition value with uncertainty and novelty
        score_i = norm_qualities[i]
        
        if u_weight > 0.:
            unct_score = sigma_sum_normed * u_weight 
            score_i += (unct_score / max(1., np.sum(sigma_sum_normed))) 
            
        normalized_novelty_factor = norm_novelty[i]  
            
        # Boost scores for novel candidates
        final_score = score_i + 0.3 * normalized_novelty_factor
        
        if campaign_progress > 0.5:
            # Apply a soft cap to prevent overfitting early on when exploration is key.
            capped_final_score = min(1., max(.2, final_score))
        else: 
             capped_final_score = final_score
            
        scores.append(capped_final_score)
    
    return np.array(scores)