def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a directional momentum term based on recent outcome trends."""
    if len(context["Y_obs"]) < 4:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    n = len(context["Y_obs"])
    split_idx = n // 2
    older_half = np.mean(context["Y_obs"][:split_idx], axis=0)
    newer_half = np.mean(context["Y_obs"][split_idx:], axis=0)
    momentum_direction = newer_half - older_half
    
    # Avoid division by zero if no movement recently
    direction_norm_sq = np.dot(momentum_direction, momentum_direction)
    if direction_norm_sq < 1e-8:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    scores = []
    names = context["objective_names"]
    for cand in context["pool"]:
        gp_mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        dot_product = np.dot(gp_mean_vec - newer_half, momentum_direction)
        normalized_dot = dot_product / np.sqrt(direction_norm_sq)
        scores.append(cand["acq_value_norm"] + 0.1 * normalized_dot)

    return scores