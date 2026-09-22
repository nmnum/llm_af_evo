def score_pool(context):
    """Score candidates based on how far their predicted objectives are from any previously observed point in objective space, rewarding exploration of untried regions regardless of predicted quality."""
    scores = []
    y_obs = context["Y_obs"]
    
    for cand in context["pool"]:
        mu = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
        
        # Compute distances to all observed points
        dists = np.linalg.norm(y_obs - mu, axis=1)
        
        # Use the minimum distance as novelty score (higher is better)
        scores.append(np.min(dists))
    
    return scores