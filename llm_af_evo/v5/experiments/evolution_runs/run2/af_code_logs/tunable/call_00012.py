def score_pool(context):
    """Integrate momentum-based guidance with acquisition value and novelty to improve exploration-exploitation balance."""
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
    X_obs = context["X_obs"] 
    for i, cand in enumerate(context["pool"]):
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Momentum term
        dot_product = np.dot(gp_mean - newer_half, momentum_direction)
        normalized_dot = dot_product / direction_norm
        
        # Novelty penalty: reduce score if candidate is close to observed points  
        cand_x = cand["x"]
        distances = np.linalg.norm(X_obs - cand_x, axis=1) 
        min_distance = np.min(distances)
        
        novelty_penalty = 0.5 * (min_distance / max(1e-8, np.std(X_obs))) if len(X_obs) > 0 else 0
        
        # Combine acquisition value with momentum and penalty
        score = cand['acq_value_norm'] + 0.2 * normalized_dot - novelty_penalty 
        scores.append(score)
    
    return scores