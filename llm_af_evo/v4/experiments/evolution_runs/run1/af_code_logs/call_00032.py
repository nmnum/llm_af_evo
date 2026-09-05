def score_pool(context):
    """Reward candidates based on how far their predicted objectives are from any previously observed point in objective space."""
    names = context["objective_names"]
    y_obs = context["Y_obs"]  # all observations so far
    
    scores = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        if len(y_obs) == 0:
            score = float('inf') 
        else:
            distances = np.linalg.norm(y_obs - gp_mean, axis=1)
            nearest_distance = np.min(distances)
            # Favor candidates with larger distance to observed points
            score =nearest_distance
            
        scores.append(score)

    return scores