def score_pool(context):
    """Score candidates by inverse of their nearest observed neighbor in objective space to encourage exploration of untried regions."""
    names = context["objective_names"]
    y_obs = context["Y_obs"]
    scores = []
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to all observed points
        dists = np.linalg.norm(y_obs - gp_mean, axis=1)
        
        # Use the minimum distance as novelty score (larger is better)
        min_dist = np.min(dists) if len(dists) > 0 else float('inf')
        
        # Invert to make larger distances more preferred
        scores.append(1.0 / (min_dist + 1e-8))
    
    return scores