def score_pool(context):
    """Blend acquisition value with novelty and momentum-aware objective alignment."""
    if len(context["Y_obs"]) < 4:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    n = len(context["Y_obs"])
    mid = n // 2
    older_half = np.mean(context["Y_obs"][:mid], axis=0)
    newer_half = np.mean(context["Y_obs"][mid:], axis=0)
    momentum_direction = newer_half - older_half
    
    # Avoid division by zero if no recent movement
    direction_norm = np.linalg.norm(momentum_direction)
    if direction_norm < 1e-8:
        return [cand['acq_value_norm'] for cand in context["pool"]]
    
    scores = []
    names = context["objective_names"]
    front_range = context["pareto_front_range"]

    # Compute novelty as inverse distance to nearest observed point
    X_obs = context["X_obs"] 
    if len(X_obs) > 0:
        distances = np.min([np.linalg.norm(cand['x'] - x_obs) for x_obs in X_obs])
        novelty_score = 1. / (distances + 1e-8)
    else:  
        novelty_score = 0.

    # Compute momentum alignment
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        dot_product = np.dot(gp_mean - newer_half, momentum_direction) 
        normalized_dot = dot_product / direction_norm
        
        acq_val = cand['acq_value_norm']
        
        # Blend acquisition value with alignment and novelty
        score = 0.7 * acq_val + 0.25 * (normalized_dot + 1.) * novelty_score + 0.05 * np.sum(gp_mean)
        scores.append(score)

    return scores