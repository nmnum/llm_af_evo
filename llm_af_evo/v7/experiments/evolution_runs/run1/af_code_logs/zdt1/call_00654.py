def modifier(context):
    """Momentum bonus: reward candidates whose predicted objectives align with recent improvement direction."""
    if len(context["Y_obs"]) < 4:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    n = len(context["Y_obs"])
    split_idx = n // 2
    
    older_means = np.mean(context["Y_obs"][:split_idx], axis=0)
    newer_means = np.mean(context["Y_obs"][split_idx:], axis=0)
    
    momentum_direction = newer_means - older_means
    norm_momentum = np.linalg.norm(momentum_direction)
    
    if norm_momentum < 1e-8:
        return [0.0] * len(context["pool"])
        
    bonus_factor = min(2.0, max(0.5, 1.0 / (norm_momentum + 1e-6)))
    values = []
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        diff_from_newer = gp_mean - newer_means
        projection = np.dot(diff_from_newer, momentum_direction)
        bonus = bonus_factor * (projection / norm_momentum)
        values.append(bonus)
        
    return values