def score_pool(context):
    """Blend acquisition value with a momentum term tracking recent improvement direction."""
    if len(context["Y_obs"]) < 4:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    names = context["objective_names"]
    n_half = len(context["Y_obs"]) // 2
    older_means = np.mean(context["Y_obs"][:n_half], axis=0)
    newer_means = np.mean(context["Y_obs"][n_half:], axis=0)
    momentum_direction = newer_means - older_means
    
    # Avoid division by zero if direction is near-zero
    dir_norm = np.linalg.norm(momentum_direction)
    if dir_norm < 1e-8:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    scores = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        dot_product = np.dot(gp_mean - newer_means, momentum_direction)
        normalized_score = dot_product / dir_norm
        # Add as a small secondary term to acquisition value
        scores.append(cand["acq_value_norm"] + 0.1 * normalized_score)
    
    return scores