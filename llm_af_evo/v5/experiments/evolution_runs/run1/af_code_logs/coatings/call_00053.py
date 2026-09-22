def score_pool(context):
    """Score candidates by normalized acquisition value enhanced with a progress-sensitive uncertainty term that adapts exploration intensity based on recent optimization dynamics."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    
    # Early termination: if not enough observations, use just acq_value_norm
    if len(context['Y_obs']) < 4:
        for cand in context["pool"]:
            scores.append(cand["acq_value_norm"])
        return scores

    # Compute momentum direction from Y_obs (split into older and newer halves)
    n = len(context['Y_obs'])
    mid_idx = n // 2
    older_half = np.mean(context['Y_obs'][:mid_idx], axis=0)
    newer_half = np.mean(context['Y_obs'][mid_idx:], axis=0)
    
    momentum_direction = newer_half - older_half
    
    # Avoid division by zero if no movement detected recently
    mom_norm = np.linalg.norm(momentum_direction) + 1e-9

    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_total = sum(gp[name]["std"] / front_range[name] for name in names)

        # Normalize candidate mean vector and compute dot product with momentum direction
        normalized_mu = np.array([gp[name]["mean"] for name in names])
        
        # Adjust uncertainty based on how much the campaign has been progressing recently:
        progress_factor = max(0.5, 1 - context["campaign"]["progress"])
        adjusted_sigma = sigma_total * (2 + progress_factor)

        score_component = cand['acq_value_norm'] 
        bonus_term = np.dot(normalized_mu, momentum_direction) / mom_norm

        # Blend acquisition value with a scaled uncertainty and the temporal signal
        final_score = 0.75 * score_component + 0.15 * adjusted_sigma + 0.10 * max(0., bonus_term)
        
        scores.append(final_score)

    return scores