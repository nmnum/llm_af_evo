def modifier(context):
    """Reward candidates aligned with recent improvement direction in objective space."""
    if len(context["Y_obs"]) < 4:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    n_half = len(context["Y_obs"]) // 2
    older_Y = context["Y_obs"][:n_half]
    newer_Y = context["Y_obs"][n_half:]
    
    older_mean = np.mean(older_Y, axis=0)
    newer_mean = np.mean(newer_Y, axis=0)
    
    momentum_direction = newer_mean - older_mean
    dir_norm = np.linalg.norm(momentum_direction)
    
    if dir_norm < 1e-8:
        return [0.0] * len(context["pool"])
        
    values = []
    for cand in context["pool"]:
        gp_means = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        diff_from_newer = gp_means - newer_mean
        projection = np.dot(diff_from_newer, momentum_direction) / dir_norm
        
        # Scale by a bounded factor (e.g., 0.5)
        values.append(0.5 * max(projection, 0.0))
    
    return values