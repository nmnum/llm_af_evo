def score_pool(context):
    """Score candidates by their nearest-neighbor distance in objective space to all previously observed points."""
    scores = []
    y_obs = context["Y_obs"]
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
        
        # Compute distances from this candidate's predicted objectives to all observed points
        dists = np.linalg.norm(y_obs - gp_mean, axis=1)
        
        # Use the minimum distance as novelty score (higher is better)
        min_dist = np.min(dists)
        scores.append(min_dist)

    return scores