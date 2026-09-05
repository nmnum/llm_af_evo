def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with an uncertainty bonus scaled by progress, and penalize nearby candidates for diversity."""
    if not context["pool"]:
        return []
    
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []

    # Compute base score as acq_value_norm plus a UCB-style uncertainty term
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        
        ucb_bonus = 0.1 * (1 - progress) * sum(cand["gp_posterior"][name]["std"] for name in names)
        base_score = acq + ubc_bonus
        
        # Apply a diversity penalty based on proximity to already selected candidates
        x_cand = cand['x']
        
        min_dist_x = float('inf')
        if len(context["X_obs"]) > 0:
            distances = np.linalg.norm(context["X_obs"] - x_cand, axis=1)
            min_dist_x = np.min(distances)

        # Avoid division by zero and apply a simple exponential decay
        diversity_penalty = 1.0 / (min_dist_x + 1e-8) if min_dist_x < float('inf') else 1.0
        
        scores.append(base_score * diversity_penalty)
    
    return scores