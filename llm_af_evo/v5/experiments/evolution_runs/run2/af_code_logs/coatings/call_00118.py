def score_pool(context):
    """Blend acquisition value with progress-adaptive uncertainty and inverse novelty penalty to balance exploration-exploitation dynamically."""
    
    names = context["objective_names"]
    campaign = context["campaign"] 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Progress-aware UCB bonus: scale uncertainty weight based on how far we are from the end
    progress = campaign["progress"]
    ucb_weight = 0.5 * (1 - min(1., max(0., progress) ** 2))
    
    front_range = context["pareto_front_range"] 
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_sum_normed = sum(gp_posterior[name]["std"]/front_range[name] for name in names)
        unc_scores.append(ucb_weight * (sigma_sum_normed / len(names)))

    # Compute novelty penalty: inverse of distance to nearest observed point
    X_obs = context["X_obs"]
    nov_penalties = []
    
    if len(X_obs) > 0:
        cand_xs = np.array([cand['x'] for cand in context["pool"]])
        dists_to_all = np.sum((X_obs[:, None] - cand_xs[None, :])**2, axis=2)
        min_dists = np.min(dists_to_all, axis=0) 
        norm_min_dist = min_dists / (len(cand_xs[0]) * max(1., campaign["stagnant_batches"]))
        
        # Inverse novelty penalty: higher reward for less explored regions
        nov_penalties = 0.35 * np.sqrt(norm_min_dist)
    else:
        nov_penalties = [0.] * len(context["pool"])

    final_scores = acq_values + np.array(unc_scores) - np.array(nov_penalties)

    return list(final_scores)