def score_pool(context):
    """
    Exploitation with novelty penalty: rank by predicted objective sum,
    then penalize candidates that are too close to already-observed points.
    This encourages exploration while maintaining strong exploitation focus.
    """
    names = context["objective_names"]
    scores = []
    
    # Compute base exploit score (sum of means)
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Add novelty penalty: inverse distance to nearest observed point
        x_cand = cand["x"]
        if len(context["X_obs"]) > 0:
            distances = np.linalg.norm(context["X_obs"] - x_cand, axis=1)
            min_distance = np.min(distances) 
            # Inverse relationship: closer points get more penalty  
            novelty_penalty = 1.0 / (min_distance + 1e-8)
        else:
            novelty_penalty = 1.0
            
        scores.append(mu_sum * novelty_penalty)

    return scores