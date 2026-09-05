def score_pool(context):
    """Estimate posterior-dominated likelihood and suppress candidates with high uncertainty in regions already well-covered by observations."""
    
    names = context["objective_names"]
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute distances from all previously observed points
    X_obs = context["X_obs"]
    if len(X_obs) == 0:
        novelty_distances = [np.inf] * len(context["pool"])
    else:
        novelty_distances = []
        for i, cand in enumerate(context["pool"]):
            x_cand = np.array(cand['x'])
            dists_to_observed = [np.linalg.norm(x_cand - x_obs) for x_obs in X_obs]
            min_dist = min(dists_to_observed)
            novelty_distances.append(min_dist)

    # Normalize distances to scale by feature dimensionality
    if len(X_obs) > 0:
        max_novelty_distance = np.max(novelty_distances)
        norm_dists = [d / (max_novelty_distance + 1e-8) for d in novelty_distances]
    else:
        norm_dists = [0.0] * len(context["pool"])

    # Estimate how much uncertainty each candidate has, normalized by front range
    sigma_sums_normed = []
    for cand in context["pool"]:
        gp_posterior = cand['gp_posterior']
        sig_sum = sum(gp_posterior[name]["std"] / context["pareto_front_range"][name] 
                      for name in names)
        sigma_sums_normed.append(sig_sum)

    # Adjust acquisition by uncertainty and proximity to existing points
    scores = []
    
    front_size = len(context.get("pareto_front", []))
    if not front_size:
        front_size = 1.0

    for i, cand in enumerate(context["pool"]):
        
        acq_score = acq_values[i]
        sigma_sum_normed = sigma_sums_normed[i] 
                
        # If candidate is too close to an observed point (within a threshold), reduce its score
        proximity_penalty_weight = 0.1 * min(1., norm_dists[i])
        
        # Reduce acquisition value for candidates that are uncertain and in covered regions  
        adjusted_score = acq_score - sigma_sum_normed / front_size - proximity_penalty_weight
        
        scores.append(adjusted_score)
    
    return scores