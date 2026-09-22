def score_pool(context):
    """Blend acquisition value with a directional momentum term based on recent objective trends."""
    if len(context["Y_obs"]) < 4:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    n = len(context["Y_obs"])
    mid = n // 2
    older_half = np.mean(context["Y_obs"][:mid], axis=0)
    newer_half = np.mean(context["Y_obs"][mid:], axis=0)
    momentum_direction = newer_half - older_half
    
    if np.linalg.norm(momentum_direction) < 1e-8:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    scores = []
    names = context["objective_names"]
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        dot_product = np.dot(gp_mean - newer_half, momentum_direction)
        normalized_dot = dot_product / np.linalg.norm(momentum_direction)
        scores.append(cand['acq_value_norm'] + 0.1 * normalized_dot)
    
    return scores