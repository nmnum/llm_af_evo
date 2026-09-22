def score_pool(context):
    """Suppress scores of candidates that are too similar in feature space to already selected ones, encouraging diversity."""
    
    names = context["objective_names"]
    pool_size = len(context["pool"])
    X_obs = context["X_obs"] 
    campaign = context["campaign"]

    # Get acquisition values as base score
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # For each candidate, compute how close it is to existing observations  
    scores = []
    for i, cand in enumerate(context["pool"]):
        x_cand = cand["x"]
        
        if len(X_obs) > 0:
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)
            
            # If candidate is very close to an existing point (within a threshold), 
            # suppress its score significantly
            feature_threshold = 0.05 * len(x_cand)  
            if min_dist < feature_threshold:
                suppression_factor = 1e6 * (feature_threshold - min_dist)
                scores.append(acq_values[i] / max(1., suppression_factor))
            else:
                # No suppression, use base acquisition value
                scores.append(float(acq_values[i]))
        else: 
            scores.append(float(acq_values[i]))

    return scores