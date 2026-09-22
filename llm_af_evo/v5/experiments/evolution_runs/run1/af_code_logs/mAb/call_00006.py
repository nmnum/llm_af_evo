def score_pool(context):
    """Blend acquisition value with progress-aware uncertainty and novelty to balance exploration vs exploitation."""
    names = context["objective_names"]
    
    # Extract base scores (already hypervolume improvement estimates)
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute normalized uncertainties
    front_range = context["pareto_front_range"]
    stds_normalized = []
    for cand in context["pool"]:
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                        for name in names)
        stds_normalized.append(sigma_sum)

    # Normalize uncertainties to [0, 1]
    min_std, max_std = np.min(stds_normalized), np.max(stds_normalized)  
    if abs(max_std - min_std) < 1e-9:
        norm_stds = np.ones_like(stds_normalized)
    else:
        norm_stds = (np.array(stds_normalized) - min_std) / (max_std - min_std + 1e-9)

    # Compute novelty scores based on distance to observed points
    X_obs = context["X_obs"]
    cand_xs = np.stack([cand['x'] for cand in context["pool"]])
    
    distances_sq = np.sum((cand_xs[:, None, :] - X_obs[None, :, :]) ** 2, axis=2)
    min_distances = np.min(distances_sq, axis=1)

    # Normalize novelty to [0, 1] range
    if len(min_distances) > 1:
        min_dist_normed = (min_distances - np.min(min_distances)) / (
            np.max(min_distances) - np.min(min_distances) + 1e-9)
    else: 
        min_dist_normed = np.zeros_like(min_distances)

    # Combine components with progress-dependent weights
    campaign_progress = context["campaign"]["progress"]
    
    if campaign_progress < 0.3:
        exploit_weight, explore_weight, novelty_weight = 0.85, 0.12, 0.03  
    elif campaign_progress < 0.6: 
        exploit_weight, explore_weight, novelty_weight = 0.75, 0.19, 0.06
    else:
        exploit_weight, explore_weight, novelty_weight = 0.40, 0.28, 0.32

    # Final score is weighted combination of all components  
    scores = (exploit_weight * acq_scores + 
              explore_weight * norm_stds +
              novelty_weight * (1 - min_dist_normed))
              
    return list(scores)