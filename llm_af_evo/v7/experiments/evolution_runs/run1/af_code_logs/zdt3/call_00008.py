def modifier(context):
    """Reward candidates whose predicted objectives align with recent improvement direction."""
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
    direction_norm = np.linalg.norm(momentum_direction)
    
    if direction_norm < 1e-8:
        return [0.0] * len(context["pool"])
        
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        pred_means = np.array([gp[name]["mean"] for name in names])
        diff_from_newer = pred_means - newer_mean
        dot_product = np.dot(diff_from_newer, momentum_direction)
        normalized_score = dot_product / direction_norm
        values.append(normalized_score * 0.5)  
    return values