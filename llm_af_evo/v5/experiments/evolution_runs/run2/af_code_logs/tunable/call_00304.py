def score_pool(context):
    """Reward candidates with large nearest-neighbor distances in objective space to previously observed points."""
    scores = []
    y_obs = context["Y_obs"]
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
        
        # Compute distance from this candidate's predicted objectives to all observed points
        distances_sq = np.sum((y_obs - gp_mean) ** 2, axis=1)
        min_distance_sq = np.min(distances_sq)
        
        # Use inverse of squared distance as score (higher is better), avoiding division by zero
        if min_distance_sq == 0:
            score = float('inf')
        else:
            score = 1.0 / min_distance_sq
            
        scores.append(score)

    return scores