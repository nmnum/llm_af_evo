def score_pool(context):
    """Blend acquisition value with uncertainty-aware progress sensitivity and inverse novelty."""
    names = context["objective_names"]
    
    # Use the provided normalized acq_value_norm directly as quality signal  
    qualities = [cand['acq_value_norm'] for cand in context["pool"]]
        
    # Normalize quality to [0, 1] 
    q_min, q_max = min(qualities), max(qualities)
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.array([0.5] * len(qualities))
    else:
        norm_qualities = (np.array(qualities) - q_min) / (q_max - q_min)

    # Compute uncertainty as sum of stds normalized by front range
    front_range = context["pareto_front_range"]
    
    uncertainties = []
    for cand in context["pool"]:
        u_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                    for name in names)
        uncertainties.append(u_sum)

    # Progress-aware uncertainty weighting: increase weight on std as campaign progresses
    progress = context['campaign']['progress']
    
    # Dynamic blend of quality and normalized uncertainty  
    w_uncertainty = 0.3 + 0.7 * (1 - np.exp(-5*progress)) 
    scores_raw = []
        
    for i, q_norm in enumerate(norm_qualities):
        u_norm = uncertainties[i] / max(uncertainties) if max(uncertainties) > 0 else 0
        score_i = w_uncertainty * u_norm + (1 - w_uncertainty) * q_norm 
        scores_raw.append(score_i)
    
    # Inverse novelty: reward candidates far from existing observations  
    x_observed = context["X_obs"]
    if len(x_observed) > 0:
        
        pool_x = np.array([cand['x'] for cand in context["pool"]])
            
        # Compute min distance to any observed point
        dists_to_obs = []
        for i, xi in enumerate(pool_x):
            distances = [np.sum((xi - xj)**2) ** 0.5 for xj in x_observed]
            mindist = np.min(distances)
            dists_to_obs.append(mindist)

        # Normalize min-dist to [0,1] and invert 
        d_min, d_max = (min(dists_to_obs), max(dists_to_obs))
        
        if abs(d_max - d_min) < 1e-9:
            novelty_scores = np.array([0.5]*len(pool_x))  
        else:    
            norm_dists = (np.array(dists_to_obs)-d_min)/(d_max-d_min)
            # Invert so large distance -> high score
            novelities = 1 - norm_dists
            
        scores_final = []
        
        for i, s_raw in enumerate(scores_raw):
            
            novelty_boost = max(0., novelities[i] * 2.5) 
                
            combined_score_i = (s_raw + novelty_boost)/2.
                    
            scores_final.append(combined_score_i)
    else:
       # No observed points yet
        scores_final = list(scores_raw)

        
    return scores_final