def score_pool(context):
    """Integrate acquisition value with an entropy-based exploration bonus and dynamic novelty penalty to encourage diverse front coverage."""
    
    names = context["objective_names"]
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    X_obs = context["X_obs"]
    campaign = context["campaign"]

    # Compute entropy-based exploration bonus from GP posteriors
    entropies = []
    front_range = context["pareto_front_range"]
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        H = 0.0
        
        # Entropy of each objective's posterior (assuming Gaussian)
        for name in names:
            std = gp_posterior[name]["std"]
            if std > 1e-8:  
                H += 0.5 * np.log(2*np.pi*np.e*std**2) 
                
        entropies.append(H)

    # Normalize entropy bonus
    max_entropy = max(entropies)
    
    if max_entropy == 0:
        entropy_bonus = [0.] * len(context["pool"])
    else:  
        entropy_bonus = np.array(entropies) / max_entropy

    # Dynamic novelty penalty based on progress and stagnation 
    stagnant_batches = campaign["stagnant_batches"]
        
    nov_rewards = []
    
    if len(X_obs) > 0:
        for cand in context["pool"]:
            x_cand = cand["x"] 
            
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            
            # Use inverse of squared distance as novelty reward
            min_dist_sq = np.min(dists**2) + 1e-8
            
            norm_min_dist = (min_dist_sq ** (-0.5)) / max(1., stagnant_batches)

            nov_rewards.append(norm_min_dist)
    else:
        nov_rewards = [0.] * len(context["pool"])

    
    # Combine terms with dynamic weights based on campaign progress
    progress = campaign["progress"]
        
    weight_acq = 0.7 + 0.3*(1 - progress)  
    weight_entropy = 0.25*progress 
    weight_novelty = 0.4 * (1 - progress**2)
    
    final_scores = (
        weight_acq * acq_values +
        weight_entropy * entropy_bonus -
        weight_novelty * np.array(nov_rewards)  
    )
        
    return list(final_scores)