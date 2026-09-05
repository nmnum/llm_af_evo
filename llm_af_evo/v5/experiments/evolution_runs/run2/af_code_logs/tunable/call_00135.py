def score_pool(context):
    """Add a momentum term based on recent improvement direction in objective space."""
    if len(context["Y_obs"]) < 4:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    n = len(context["Y_obs"])
    mid = n // 2
    older_half = np.mean(context["Y_obs"][:mid], axis=0)
    newer_half = np.mean(context["Y_obs"][mid:], axis=0)
    momentum_direction = newer_half - older_half
    
    # Avoid division by zero if there's no recent movement
    direction_norm = np.linalg.norm(momentum_direction)
    if direction_norm < 1e-8:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    scores = []
    names = context["objective_names"]
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        dot_product = np.dot(gp_mean - newer_half, momentum_direction)
        normalized_score = dot_product / direction_norm
        scores.append(cand["acq_value_norm"] + 0.1 * normalized_score)

    return scores