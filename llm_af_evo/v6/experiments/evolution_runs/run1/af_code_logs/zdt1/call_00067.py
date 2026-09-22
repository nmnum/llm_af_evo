def modifier(context):
    """Momentum bonus: reward candidates whose predicted direction aligns with recent improvement trend."""
    if len(context["Y_obs"]) < 4:
        return [0.0] * len(context["pool"])
    
    n = len(context["Y_obs"])
    split_idx = n // 2
    older_half = context["Y_obs"][:split_idx]
    newer_half = context["Y_obs"][split_idx:]
    
    names = context["objective_names"]
    older_mean = np.mean(older_half, axis=0)
    newer_mean = np.mean(newer_half, axis=0)
    momentum_direction = newer_mean - older_mean
    
    # Avoid division by zero
    direction_norm_sq = np.sum(momentum_direction ** 2)
    if direction_norm_sq < 1e-12:
        return [0.0] * len(context["pool"])
    
    values = []
    for cand in context["pool"]:
        gp_means = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        diff_from_newer_mean = gp_means - newer_mean
        dot_product = np.dot(diff_from_newer_mean, momentum_direction)
        normalized_bonus = max(0.0, min(dot_product / direction_norm_sq, 1.0))
        values.append(normalized_bonus * 0.2)  
    
    return values