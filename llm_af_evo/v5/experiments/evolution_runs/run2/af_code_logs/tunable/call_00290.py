def score_pool(context):
    """Integrate hypervolume acquisition with a front-coverage-aware uncertainty bonus and adaptive diversity reward to dynamically balance targeted exploration and progressive exploitation."""
    
    names = context["objective_names"]
    campaign = context['campaign']
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute normalized distances from candidates to Pareto front
    pareto_front = context["pareto_front"]
    ref_point = context["ref_point"]
    
    if len(pareto_front) > 0:
        dists_to_pf = []
        for i, cand in enumerate(context["pool"]):
            x_cand = np.array(cand['x'])
            
            # Find the closest point on Pareto front (Euclidean distance)
            min_dist = float('inf')
            for pf_point in pareto_front:
                dist_sq = sum((pf_point[j] - ref_point[j]) ** 2 
                              for j in range(len(names)))
                if dist_sq < min_dist:  
                    min_dist = dist_sq
            dists_to_pf.append(min_dist)
        
        # Normalize to [0,1]
        max_dist = np.max(dists_to_pf) + 1e-8   # avoid division by zero 
        pf_distances_normed = np.array(dists_to_pf)/max_dist
        
    else:
        pf_distances_normed = np.zeros(len(context["pool"]))
    
    front_range = context['pareto_front_range']
    
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_sum = sum(gp_posterior[name]["std"]/front_range[name] for name in names)
        
        # Progress-adaptive uncertainty bonus
        progress = campaign['progress']  
        ucb_weight = 1.0 - min(1., max(.5, (2 * progress) ** 3)) 
        unc_scores.append(sigma_sum * ucb_weight)

    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        
        # Compute pairwise distances among observed points
        dists_matrix = np.sum((X_obs[:, None] - X_obs[None, :])**2, axis=2)
        np.fill_diagonal(dists_matrix, float('inf'))
                
        nov_rewards = []
        for i in range(len(context["pool"])):
            cand_x = context['pool'][i]["x"]
            
            # Find minimum distance to any observed point  
            min_dist_to_observed = None
            x_cand_np = np.array(cand_x)
                        
            if len(X_obs) > 0:
                dists_from_cand = np.sum((X_obs - x_cand_np)**2, axis=1)
                
                # Compute a diversity reward inversely proportional to min distance  
                closest_dist = np.min(dists_from_cand)
            
                norm_closest = max(0., (closest_dist / 3.)) 
            else:
                 norm_closest = float('inf') 
            
            nov_rewards.append(norm_closest)    
        
        # Normalize rewards
        if len(nov_rewards):
             max_novelty_score = np.max(nov_rewards)
            
             normalized_rewds = [r/max_novelty_score  for r in nov_rewards]  
             
         else:
              normalized_rewds = []
    else: 
          normalized_rewds = []

    
    # Final scoring combining all components
    final_scores = acq_values + np.array(unc_scores) - .25 * (np.array(pf_distances_normed)) 
    
        
    return list(final_scores)