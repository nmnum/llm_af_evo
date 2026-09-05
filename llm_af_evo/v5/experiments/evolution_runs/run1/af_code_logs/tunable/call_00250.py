def score_pool(context):
    """Reward candidates based on how far their predicted objectives are from any previously observed point in objective space."""
    names = context["objective_names"]
    y_obs = context["Y_obs"]
    scores = []
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to all observed points
        dists = np.linalg.norm(y_obs - gp_mean, axis=1)
        min_dist = np.min(dists) if len(dists) > 0 else float('inf')
        
        scores.append(min_dist)

    return scores