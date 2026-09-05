def score_pool(context):
    """Balance acquisition value with uncertainty-weighted Pareto front proximity and dynamic novelty incentive."""
    
    names = context["objective_names"]
    campaign = context["campaign"] 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute normalized distance to nearest point on the current pareto front
    pf = context["pareto_front"]
    if len(pf) == 0:
        distances_to_pf = np.ones(len(context["pool"])) * float('inf')
    else:
        candidates_x = np.array([cand['x'] for cand in context["pool"]])
        
        # Compute squared Euclidean distance from each candidate to every point on PF
        expanded_PF = pf[:, None, :] 
        expanded_candidates = candidates_x[None, :, :]
        diff_sq = (expanded_PF - expanded_candidates) ** 2  
        sum_diff_sq = np.sum(diff_sq, axis=2)
        
        min_distances_squared = np.min(sum_diff_sq, axis=0)
        distances_to_pf = np.sqrt(min_distances_squared + 1e-8)

    # Normalize by the observed range of objectives
    front_range = context["pareto_front_range"]
    
    if len(pf) > 0:
        normalized_dists = []
        for dist in distances_to_pf: 
            norm_dist = min(dist / (np.mean(list(front_range.values())) + 1e-8), 1.0)
            normalized_dists.append(norm_dist)
        normalized_dists = np.array(normalized_dists)  
    else:
        normalized_dists = np.zeros(len(context["pool"]))

    # Progress-aware uncertainty scaling
    progress = campaign["progress"]
    
    ucb_weight = max(0., 1. - progress**2)

    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        unc_score = ucb_weight * sigma_sum
        unc_scores.append(unc_score)

    
    # Dynamic novelty term based on stagnation and progress  
    stagnant_batches = campaign["stagnant_batches"]
    
    X_obs = context["X_obs"]
        
    if len(X_obs) > 0:
        nov_rewards = []
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)

            # Scale novelty reward by stagnation and inverse of squared distance  
            nov_reward = (stagnant_batches + 1.) / ((min_dist**2) + 1e-8)
            
           nov_rewards.append(nov_reward)
    else:
        nov_rewards = [0.0] * len(context["pool"])

    
    # Combine all terms
    final_scores = acq_values - np.array(unc_scores) + (np.exp(-normalized_dists)) / (np.array(nov_rewards)+1e-8)

    return list(final_scores)