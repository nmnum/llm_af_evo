def score_pool(context):
    """Blend acquisition value with uncertainty and novelty to improve exploration-exploitation balance."""
    if not context["pool"]:
        return []
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    X_obs = context["X_obs"] 
    Y_obs = context["Y_obs"]

    # Base acquisition scores
    acq_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Uncertainty bonus: sum of normalized standard deviations  
    uncertainty_bonus = []
    for cand in context["pool"]:
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                        for name in names)
        uncertainty_bonus.append(sigma_sum)

    # Novelty score based on distance to existing observations
    novelty_scores = []  
    for i, cand in enumerate(context["pool"]):
        x_i = cand["x"]
        
        if len(X_obs) == 0:
            novel_score = 1.0 
        else:   
            distances = [np.linalg.norm(x_i - obs_x) for obs_x in X_obs]
            min_distance = np.min(distances)
            
            # Normalize by the range of observed features
            feature_range = np.max(X_obs, axis=0) - np.min(X_obs, axis=0) 
            if not np.any(feature_range == 0):
                normalized_dist = min_distance / np.mean(feature_range)
            else:
                normalized_dist = float('inf') 

            # Invert so higher distance means more novel
            novelty_score = max(1e-6, 1.0 - normalized_dist) 
                
        novelty_scores.append(novelty_score)

    # Combine: weighted sum of acquisition value + uncertainty bonus + novelty  
    alpha_acq = 0.7   # weight for base acqusition score
    beta_uncert = 0.25  # weight for uncertainty   
    gamma_novely = 0.05  # weight for novelty
    
    scores = []
    for i in range(len(context["pool"])):
        combined_score = (alpha_acq * acq_scores[i] + 
                          beta_uncert * uncertainty_bonus[i] +
                          gamma_novely * novelty_scores[i])
        
        scores.append(combined_score)
    
    return scores