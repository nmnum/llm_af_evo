def score_pool(context):
    """Score candidates by blending acquisition value with an adaptive entropy-based diversity reward that emphasizes underexplored regions of objective space."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute a baseline novelty term based on distance to the nearest previously observed point
    if len(context['X_obs']) == 0:
        min_distances = np.full(len(context["pool"]), fill_value=np.inf)
    else:
        X_obs = context['X_obs']
        distances = []
        for cand in context["pool"]:
            x_cand = cand["x"]
            dists_to_observed = np.linalg.norm(X_obs - x_cand, axis=1)
            min_dist = np.min(dists_to_observed)
            distances.append(min_dist)
        min_distances = np.array(distances)

    # Normalize the novelty term to [0, 1] range
    if len(context['X_obs']) > 0:
        max_distance = np.max(np.linalg.norm(X_obs[:, None] - X_obs[None], axis=2))
        novel_scores = (max_distance - min_distances) / max_distance if max_distance != 0 else np.zeros_like(min_distances)
    else:
        novel_scores = np.ones(len(context["pool"]))

    # Entropy-based diversity: quantify how much uncertainty is spread across objectives
    entropy_rewards = []
    for cand in context["pool"]:
        gp_posterior = cand['gp_posterior']
        
        stds_normalized_by_range = [gp_posterior[name]["std"]/front_range[name] 
                                    for name in names]
        # Compute the geometric mean of normalized standard deviations as a proxy
        if all(s > 0 for s in stds_normalized_by_range):
            entropy_reward = np.exp(np.mean([np.log(std) for std in stds_normalized_by_range]))
        else:
            entropy_reward = 1.0

        # Encourage high uncertainty spread (i.e., low correlation among objectives)
        if len(names) > 2:  
             cov_matrix_diag_sum = sum(gp_posterior[name]["std"]**2 
                                       for name in names)

             variances_product_root = np.prod([gp_posterior[name]["std"]
                                                for name in names])**(
                                                  1.0 / len(names))
            
            # Reward candidates with more spread-out uncertainties
            entropy_reward *= (cov_matrix_diag_sum /
                               max(variances_product_root, 1e-8))

        entropy_rewards.append(entropy_reward)

    weights = np.array([np.log(e + 1) for e in entropy_rewards])
    
    scores = []
    alpha_explore = context['campaign']['progress'] ** 2
    beta_novelty   = (0.5 * max(context["campaign"]["stagnant_batches"] - 3, 0)) / (
            len(names)*max(1e-6,
                           np.std([gp_posterior[name]["std"]
                                   for name in names 
                                   for gp_posterior in [c['gp_posterior']  
                                                       for c in context["pool"]] ])
                          ) + 5.0)

    beta_novelty = min(beta_novelty, 1.)

    
    # Final score combining acquisition value with entropy diversity and normalized novelty
    acq_norms   = np.array([cand['acq_value_norm'] 
                            for cand in context["pool"]])
    
   
    adjusted_scores= (alpha_explore * acq_norms +  
                      beta_novelty  *(novel_scores) +
                       weights)

    return list(adjusted_scores)