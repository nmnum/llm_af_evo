def score_pool(context):
    """Score candidates by inverse distance to nearest observed point in objective space."""
    names = context["objective_names"]
    y_obs = context["Y_obs"] 
    scores = []
    
    for cand in context["pool"]:
        mu_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        if len(y_obs) == 0:
            score = float('inf')
        else:
            distances = np.linalg.norm(y_obs - mu_vec, axis=1)
            nearest_distance = np.min(distances)
            
            # Avoid division by zero; use a small epsilon
            eps = 1e-8  
            if nearest_distance < eps: 
                score = float('inf')
            else:
                score = 1.0 / nearest_distance
                
        scores.append(score)

    return scores