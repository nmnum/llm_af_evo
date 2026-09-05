def score_pool(context):
    """Add a momentum term based on recent improvement direction in objective space."""
    if len(context["Y_obs"]) < 4:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    names = context['objective_names']
    half = len(context["Y_obs"]) // 2
    older_mean = np.mean(context["Y_obs"][:half], axis=0)
    newer_mean = np.mean(context["Y_obs"][half:], axis=0)
    momentum_direction = newer_mean - older_mean
    
    # Avoid division by zero if there's no recent movement
    direction_norm = np.linalg.norm(momentum_direction)
    if direction_norm < 1e-8:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    scores = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        deviation_from_newer = gp_mean - newer_mean
        momentum_score = np.dot(deviation_from_newer, momentum_direction) / direction_norm
        scores.append(cand["acq_value_norm"] + 0.1 * momentum_score)
    
    return scores