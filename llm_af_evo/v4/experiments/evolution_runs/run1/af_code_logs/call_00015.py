def score_pool(context):
    """Blend acquisition value with uncertainty and diversity penalty to improve exploration while maintaining batch quality."""
    names = context["objective_names"]
    scores = []
    
    # Base score from acq_value_norm
    base_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Add UCB-style bonus for uncertainty (scaled by progress)
    ucb_bonus = [
        0.1 * sum(cand["gp_posterior"][name]["std"] for name in names) 
        for cand in context["pool"]
    ]
    
    # Compute diversity penalty based on proximity to already selected candidates
    X_obs = context["X_obs"]
    if len(X_obs) > 0:
        distances = np.array([
            min(np.linalg.norm(cand['x'] - x_obs) for x_obs in X_obs)
            for cand in context["pool"] 
        ])
        
        # Normalize distance to [0,1] range (assuming max possible dist is ~sqrt(6*2))
        normalized_distances = distances / np.sqrt(6 * 2)

        diversity_penalty = [
            -np.exp(-d) if d < float('inf') else 0.0 
            for d in normalized_distances
        ]
    else:
        # No previous observations, no penalty.
        diversity_penalty = [0.] * len(context["pool"])

    final_scores = np.array(base_scores) + \
                   np.array(ucb_bonus) + \
                   np.array(diversity_penalty)
    
    return list(final_scores)