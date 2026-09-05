def score_pool(context):
    """Blend acquisition value with a momentum term favoring directions recently improving."""
    if len(context["Y_obs"]) < 4:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    names = context["objective_names"]
    n_half = len(context["Y_obs"]) // 2
    older_mean = np.mean(context["Y_obs"][:n_half], axis=0)
    newer_mean = np.mean(context["Y_obs"][n_half:], axis=0)
    momentum_direction = newer_mean - older_mean
    
    # Avoid division by zero if no movement recently
    direction_norm = np.linalg.norm(momentum_direction)
    if direction_norm < 1e-8:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    scores = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        dot_product = np.dot(gp_mean - newer_mean, momentum_direction)
        normalized_dot = dot_product / direction_norm
        # Add as a small secondary term (0.1 factor) to acquisition value
        score = cand['acq_value_norm'] + 0.1 * normalized_dot
        scores.append(score)
    
    return scores