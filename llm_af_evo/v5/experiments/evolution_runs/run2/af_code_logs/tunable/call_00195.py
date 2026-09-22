def score_pool(context):
    """Add a momentum term based on recent improvement direction in objective space."""
    if len(context["Y_obs"]) < 4:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    n = len(context["Y_obs"])
    half_n = n // 2
    older_half = np.mean(context["Y_obs"][:half_n], axis=0)
    newer_half = np.mean(context["Y_obs"][half_n:], axis=0)
    momentum_direction = newer_half - older_half
    
    # Avoid division by zero if there's no recent movement
    direction_norm_sq = np.dot(momentum_direction, momentum_direction)
    if direction_norm_sq < 1e-9:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    scores = []
    names = context['objective_names']
    for cand in context["pool"]:
        gp_mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        diff_from_newer = gp_mean_vec - newer_half
        dot_product = np.dot(diff_from_newer, momentum_direction)
        normalized_momentum_score = dot_product / np.sqrt(direction_norm_sq)
        
        # Blend the new term as a small additive component to acquisition value.
        final_score = cand["acq_value_norm"] + 0.1 * normalized_momentum_score
        scores.append(final_score)

    return scores