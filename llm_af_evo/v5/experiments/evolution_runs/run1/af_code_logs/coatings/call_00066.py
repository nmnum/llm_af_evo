def score_pool(context):
    """Blend acquisition value with a momentum term favoring directions aligned to recent improvement trend."""
    if len(context["Y_obs"]) < 4:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    names = context["objective_names"]
    n_half = len(context["Y_obs"]) // 2
    older_Y, newer_Y = context["Y_obs"][:n_half], context["Y_obs"][n_half:]
    older_mean = np.mean(older_Y, axis=0)
    newer_mean = np.mean(newer_Y, axis=0)
    
    momentum_direction = newer_mean - older_mean
    
    # Avoid division by zero
    dir_norm_sq = np.dot(momentum_direction, momentum_direction)
    if dir_norm_sq < 1e-8:
        return [cand['acq_value_norm'] for cand in context["pool"]]
        
    scores = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        dot_prod = np.dot(gp_mean - newer_mean, momentum_direction)
        # Normalize by direction norm
        normalized_dot = dot_prod / np.sqrt(dir_norm_sq)
        
        score = cand['acq_value_norm'] + 0.1 * normalized_dot
        scores.append(score)
    
    return scores