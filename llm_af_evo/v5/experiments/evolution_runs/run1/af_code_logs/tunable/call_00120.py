def score_pool(context):
    """Score candidates based on acquisition value adjusted for proximity to already observed points, suppressing near-duplicates."""
    names = context["objective_names"]
    ref_point = np.array(names)  # Placeholder; will be replaced by actual reference point values if needed.
    
    scores = []
    x_observed = context['X_obs']
    
    for cand in context["pool"]:
        acq_value_norm = cand["acq_value_norm"] 
        cand_x = cand["x"]
        
        min_dist_to_obs = np.inf
        for obs_x in x_observed:
            dist = np.linalg.norm(cand_x - obs_x)
            if dist < min_dist_to_obs:
                min_dist_to_obs = dist
                
        # Suppress candidates that are too close to already observed points (near-duplicates).
        proximity_penalty = 1.0 / (1.0 + min_dist_to_obs)  
        
        score = acq_value_norm * proximity_penalty
        scores.append(score)
    
    return scores