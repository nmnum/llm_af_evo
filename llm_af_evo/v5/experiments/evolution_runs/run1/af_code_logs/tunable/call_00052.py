def score_pool(context):
    """Add a momentum term based on recent improvement direction in objective space."""
    if len(context["Y_obs"]) < 4:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    n = len(context["Y_obs"])
    half = n // 2
    older_half = context["Y_obs"][:half]
    newer_half = context["Y_obs"][half:]
    
    names = context["objective_names"]
    older_mean = np.mean(older_half, axis=0)
    newer_mean = np.mean(newer_half, axis=0)
    momentum_direction = newer_mean - older_mean
    
    # Avoid division by zero
    direction_norm = np.linalg.norm(momentum_direction)
    if direction_norm < 1e-8:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        candidate_mean_vector = np.array([gp_posterior[name]["mean"] for name in names])
        
        # Dot product of (candidate - newer mean) with momentum direction
        dot_product = np.dot(candidate_mean_vector - newer_mean, momentum_direction)
        normalized_dot = dot_product / direction_norm
        
        scores.append(cand["acq_value_norm"] + 0.1 * normalized_dot)

    return scores