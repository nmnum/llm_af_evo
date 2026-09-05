def score_pool(context):
    """Add a momentum term based on recent improvement direction in objective space."""
    if len(context["Y_obs"]) < 4:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    names = context["objective_names"]
    n_half = len(context["Y_obs"]) // 2
    older_Y = context["Y_obs"][:n_half]
    newer_Y = context["Y_obs"][n_half:]
    
    older_mean = np.mean(older_Y, axis=0)
    newer_mean = np.mean(newer_Y, axis=0)
    momentum_direction = newer_mean - older_mean
    
    # Avoid division by zero
    direction_norm_sq = np.dot(momentum_direction, momentum_direction)
    if direction_norm_sq < 1e-8:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    scores = []
    for cand in context["pool"]:
        gp_mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        deviation_from_newer = gp_mean_vec - newer_mean
        momentum_score = np.dot(deviation_from_newer, momentum_direction) / np.sqrt(direction_norm_sq)
        
        # Blend with acquisition value; keep it a small secondary term.
        score = cand['acq_value_norm'] + 0.1 * momentum_score  
        scores.append(score)

    return scores