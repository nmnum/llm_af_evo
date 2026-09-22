def score_pool(context):
    """Score candidates by inverse of their nearest observed neighbor in objective space to encourage exploring untried regions."""
    names = context["objective_names"]
    y_obs = context["Y_obs"]
    scores = []
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to all observed points
        dists = np.linalg.norm(y_obs - gp_mean, axis=1)
        
        # Use the minimum distance as novelty score (higher is better)
        min_dist = np.min(dists)
        if min_dist == 0:
            scores.append(-np.inf)  # Avoid division by zero; this candidate has been observed
        else: 
            scores.append(1. / min_dist)

    return scores