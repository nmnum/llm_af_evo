def score_pool(context):
    """Suppress scores of candidates that are too close in feature space to already-selected points, encouraging exploration of diverse regions."""
    
    names = context["objective_names"]
    campaign = context["campaign"] 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Use a decay factor based on progress and stagnation to control suppression intensity
    progress = campaign["progress"]
    stagnant_batches = campaign["stagnant_batches"]
    suppress_factor = 0.3 * (1 - progress) + 0.2 * min(stagnant_batches / 5., 1.)
    
    X_obs = context["X_obs"] 
    scores = []
    
    for i, cand in enumerate(context["pool"]):
        x_cand = cand["x"]
        
        # Compute minimum squared Euclidean distance to any observed point
        if len(X_obs) > 0:
            dists_sq = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist_sq = np.min(dists_sq)

            # Suppress score based on proximity (inverse relationship with suppression factor and squared distance)
            suppress_score = max(0., 1. - suppress_factor * min_dist_sq)  
        else:
            suppress_score = 1.
            
        scores.append(acq_values[i] * suppress_score)
        
    return list(scores)