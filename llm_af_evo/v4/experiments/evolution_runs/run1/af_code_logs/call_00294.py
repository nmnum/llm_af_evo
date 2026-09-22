def score_pool(context):
    """Incorporate temporal momentum by rewarding candidates aligned with recent improvement direction."""
    if len(context["Y_obs"]) < 4:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    n = len(context["Y_obs"])
    split_idx = n // 2
    older_half = np.mean(context["Y_obs"][:split_idx], axis=0)
    newer_half = np.mean(context["Y_obs"][split_idx:], axis=0)
    momentum_direction = newer_half - older_half
    
    # Avoid division by zero if no recent movement
    direction_norm = np.linalg.norm(momentum_direction)
    if direction_norm < 1e-8:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    scores = []
    names = context["objective_names"]
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        deviation_from_newer = gp_mean - newer_half
        momentum_score = np.dot(deviation_from_newer, momentum_direction) / direction_norm
        total_score = cand['acq_value_norm'] + 0.1 * momentum_score  
        scores.append(total_score)
    return scores