def score_pool(context):
    """Reward candidates with large nearest-neighbor distances in objective space to previously observed points."""
    names = context["objective_names"]
    y_obs = context["Y_obs"]
    scores = []
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances from this candidate's predicted objectives to all observed points
        dists = np.linalg.norm(y_obs - gp_mean, axis=1)
        min_dist = np.min(dists)
        
        scores.append(min_dist)

    return scores