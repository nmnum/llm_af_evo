def score_pool(context):
    """Score by acquisition value reduced by a repulsion term that penalizes nearby previously observed points."""
    if not context["pool"]:
        return []
    
    scores = [cand["acq_value_norm"] for cand in context["pool"]]
    x_obs = context["X_obs"]
    
    # For each candidate, compute how much it's near any already-observed point
    for i, cand in enumerate(context["pool"]):
        min_dist = float('inf')
        
        # Find distance to nearest previously observed sample (excluding self if present)
        x_cand = cand['x']
        for obs_x in x_obs:
            dist = np.linalg.norm(x_cand - obs_x) 
            if dist < min_dist:
                min_dist = dist
        
        # Apply repulsion: reduce score by inverse distance
        if min_dist > 0 and not np.isinf(min_dist):
            scores[i] -= 1.5 * (1 / min_dist)
    
    return scores