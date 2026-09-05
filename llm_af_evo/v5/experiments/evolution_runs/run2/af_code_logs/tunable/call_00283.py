def score_pool(context):
    """Add a momentum term based on recent improvement direction in objective space."""
    if len(context["Y_obs"]) < 4:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    names = context["objective_names"]
    n_half = len(context["Y_obs"]) // 2
    older_y = np.mean(context["Y_obs"][:n_half], axis=0)
    newer_y = np.mean(context["Y_obs"][n_half:], axis=0)
    momentum_direction = newer_y - older_y
    
    # Avoid division by zero if no movement recently
    direction_norm_sq = np.dot(momentum_direction, momentum_direction)
    if direction_norm_sq < 1e-8:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    scores = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        diff_from_newer = gp_mean - newer_y
        momentum_score = np.dot(diff_from_newer, momentum_direction) / np.sqrt(direction_norm_sq)
        scores.append(cand['acq_value_norm'] + 0.1 * momentum_score)
    
    return scores