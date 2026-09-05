def score_pool(context):
    """Suppress scores of near-duplicate candidates based on feature proximity and blend with acquisition value for diverse exploration."""
    
    names = context["objective_names"]
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Build a suppression score based on candidate similarity
    X_obs = context["X_obs"]
    scores = []
    
    if len(X_obs) == 0:
        # No observations yet, use acquisition value only  
        return list(acq_values)
        
    feature_dim = X_obs.shape[1]
    obs_mean = np.mean(X_obs, axis=0)

    for cand in context["pool"]:
        x_cand = cand['x']
            
        # Compute distance to the mean of observed points (normalized by dimensionality) 
        dist_to_mean = np.linalg.norm(x_cand - obs_mean)
        
        # Normalize this relative to feature space size
        normalized_dist = dist_to_mean / max(1.0, np.sqrt(feature_dim))
                
        # Scale suppression factor: candidates closer to the mean get lower scores  
        suppress_factor = 2 * (normalized_dist ** (-3)) if normalized_dist > 0 else float('inf')
        
        base_score = acq_values[0] if len(acq_values) == 1 else np.mean([acq_values[i] for i in range(len(context["pool"]))])
            
        # Apply suppression
        score_with_suppression = max(0.0, base_score - suppress_factor * (1 / (len(X_obs)+1)))
        
        scores.append(score_with_suppression)
    
    return scores