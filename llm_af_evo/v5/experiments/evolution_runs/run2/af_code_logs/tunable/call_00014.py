def score_pool(context):
    """Blend acquisition value with novelty and uncertainty-aware diversity to improve exploration-exploitation balance."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Base scores from hypervolume improvement estimate  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Novelty: inverse distance to nearest previously observed point
    X_obs = context["X_obs"]
    novelty_scores = []
    if len(X_obs) > 0:
        for cand in context["pool"]:
            distances = np.linalg.norm(X_obs - cand['x'], axis=1)
            min_distance = np.min(distances)
            # Inverse of distance (higher is better), with small epsilon to avoid inf
            novelty_scores.append(1. / (min_distance + 1e-8))
    else:
        novelty_scores = [0.] * pool_size

    # Uncertainty: sum of normalized standard deviations  
    front_range = context["pareto_front_range"]
    uncertainty_scores = []
    for cand in context["pool"]:
        sigma_sum_norm = sum(cand['gp_posterior'][name]["std"] / front_range[name] 
                             for name in names)
        uncertainty_scores.append(sigma_sum_norm)

    # Combine: weighted blend of acquisition, novelty and uncertainty
    w_acq = 0.7  
    w_novelty = 0.15
    w_uncertainty = 0.15
    
    final_scores = (w_acq * acq_scores + 
                    w_novelty * np.array(novelty_scores) +
                    w_uncertainty * np.array(uncertainty_scores))
    
    return list(final_scores)