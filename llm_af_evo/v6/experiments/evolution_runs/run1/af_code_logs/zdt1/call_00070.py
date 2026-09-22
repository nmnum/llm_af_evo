def modifier(context):
    """Reward candidates whose predicted objectives align with recent improvement direction."""
    if len(context["Y_obs"]) < 4:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    n = len(context["Y_obs"])
    split_idx = n // 2
    
    older_mean = np.mean(context["Y_obs"][:split_idx], axis=0)
    newer_mean = np.mean(context["Y_obs"][split_idx:], axis=0)
    
    momentum_direction = newer_mean - older_mean
    norm_momentum = np.linalg.norm(momentum_direction)
    
    if norm_momentum < 1e-8:
        return [0.0] * len(context["pool"])
        
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        pred_means = np.array([gp[name]["mean"] for name in names])
        diff_from_newer = pred_means - newer_mean
        dot_product = np.dot(diff_from_newer, momentum_direction)
        normalized_score = dot_product / norm_momentum
        
        # Scale by a bounded factor (e.g., 0.5) to prevent over-amplification.
        values.append(0.5 * normalized_score)

    return values