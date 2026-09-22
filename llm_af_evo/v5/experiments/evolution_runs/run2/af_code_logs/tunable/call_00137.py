def score_pool(context):
    """Blend acquisition value with a momentum term tracking recent improvement direction."""
    if len(context["Y_obs"]) < 4:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    n = len(context["Y_obs"])
    split_idx = n // 2
    older_half = np.mean(context["Y_obs"][:split_idx], axis=0)
    newer_half = np.mean(context["Y_obs"][split_idx:], axis=0)
    momentum_direction = newer_half - older_half
    
    # Avoid division by zero if direction is near-zero
    dir_norm = np.linalg.norm(momentum_direction)
    if dir_norm < 1e-8:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        dot_product = np.dot(gp_mean - newer_half, momentum_direction)
        normalized_dot = dot_product / dir_norm
        score = cand['acq_value_norm'] + 0.1 * normalized_dot
        scores.append(score)
    
    return scores