def score_pool(context):
    """Blend acquisition value with progress-adaptive exploitation and entropy-based novelty."""
    
    names = context["objective_names"]
    campaign = context["campaign"] 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Progress-aware exploitation: scale down reliance on uncertainty as we advance
    progress = campaign["progress"]
    exploit_weight = 0.7 * (1 - progress**3)
    
    front_range = context["pareto_front_range"]
    entropy_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        # Compute normalized uncertainty entropy across objectives
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        
        if progress < 0.3:
            # Early phase: emphasize exploration via higher uncertainty weight  
            score = (1 - exploit_weight) * sigma_sum 
        else:
            # Later phases: favor exploitation with lower uncertainty bonus
            score = exploit_weight * sigma_sum
            
        entropy_scores.append(score)

    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        
        nov_rewards = []
            
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)

            # Entropy-based novelty: higher entropy (more uniform distribution) of nearby points
            k_nearest = 5  
            
            if len(X_obs) >= k_nearest:
                nearest_indices = dists.argsort()[:k_nearest]
                
                X_nearby = X_obs[nearest_indices] 
                    
                # Compute feature-wise variance to estimate local entropy/coverage
                variances = np.var(X_nearby, axis=0)
            
                avg_variance = np.mean(variances) 
            
            else:
                 avg_variance = 1.0
                
             norm_min_dist = min_dist / (len(x_cand)) 
                
              # Normalize by variance: higher variation implies more informative regions
               novelty_score = -norm_min_dist * max(0., 2.-avg_variance)
               
                nov_rewards.append(novelty_score)  
    else:
        nov_rewards = [0.0] * len(context["pool"])

        
     final_scores = acq_values + np.array(entropy_scores) + np.array(nov_rewards)

    
   return list(final_scores)