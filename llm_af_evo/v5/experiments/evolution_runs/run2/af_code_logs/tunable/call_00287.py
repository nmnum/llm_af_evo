def score_pool(context):
    """Leverage gradient-like information from objective means to bias toward under-explored front regions while scaling uncertainty rewards based on campaign progress."""
    
    names = context["objective_names"]
    pf = context["pareto_front"] 
    ref_point = context["ref_point"]
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    campaign = context["campaign"]

    # Compute gradient-like score based on how much each candidate's mean
    # moves toward the reference point (i.e., improves hypervolume)
    grad_scores = []
    for cand in context["pool"]:
        gp_mean = np.array([cand['gp_posterior'][name]["mean"] for name in names])
        ref_improvement = np.linalg.norm(ref_point - gp_mean, ord=2) 
        # Inverse of distance to reference point (higher is better)
        grad_scores.append(1.0 / max(1e-8, ref_improvement))
    
    # Scale uncertainty bonus based on campaign progress
    ucb_weight = 0.5 * np.exp(-campaign["progress"]*2) 
    
    front_range = context["pareto_front_range"]
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        unc_scores.append(ucb_weight * sigma_sum)

    # Novelty reward: encourage exploration of new areas
    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        nov_rewards = []
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)
            
            # Normalize by feature dimensionality
            norm_min_dist = min_dist / len(x_cand)  
            nov_rewards.append(norm_min_dist)

    else:
        nov_rewards = [0.0] * len(context["pool"])

    
    final_scores = acq_values + np.array(grad_scores) - 2.*np.array(nov_rewards)
        
    return list(final_scores)