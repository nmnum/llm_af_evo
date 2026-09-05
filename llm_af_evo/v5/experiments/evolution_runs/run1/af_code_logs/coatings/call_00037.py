def score_pool(context):
    """Blend acquisition value with an uncertainty-weighted progress signal that rewards candidates trending toward under-explored front regions."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute the normalized hypervolume improvement per candidate
    scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    if len(context["Y_obs"]) < 4:
        return scores
    
    half_idx = len(context["Y_obs"]) // 2
    older_Y = np.array(context["Y_obs"][:half_idx])
    newer_Y = np.array(context["Y_obs"][half_idx:])
    
    # Compute mean of each objective for the two halves (all maximized)
    old_mean = np.mean(older_Y, axis=0)
    new_mean = np.mean(newer_Y, axis=0)

    momentum_direction = new_mean - old_mean
    
    if not any(momentum_direction):
        return scores  # no movement in direction to reward

    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand['gp_posterior']
        
        pred_means = np.array([gp_posterior[name]["mean"] for name in names])
        ref_dist_pred = ref_point - pred_means
        
        # Compute uncertainty-adjusted distance from reference point
        ucb_penalty = sum(gp_posterior[name]["std"] / (ref_point[i] - old_mean[i]) 
                          if not np.isclose(ref_point[i], old_mean[i]) else 0.0 for i, name in enumerate(names))
        
        progress_signal = max(0., momentum_direction @ pred_means) * ucb_penalty
        
        scores[i] += 0.3 * progress_signal
    
    return scores