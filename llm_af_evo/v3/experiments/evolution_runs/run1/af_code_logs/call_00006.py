def score_pool(context):
    """Rank candidates by combining normalized predicted quality with mutual diversity, avoiding negative quality penalties from similarity weighting."""
    names = context["objective_names"]
    
    # Compute raw qualities (sum of means) for each candidate
    qualities = np.array([sum(cand["gp_posterior"][name]["mean"] for name in names)
                          for cand in context["pool"]])
    
    # Normalize quality to [0, 1] range
    q_min, q_max = qualities.min(), qualities.max()
    if q_max > q_min:
        norm_qualities = (qualities - q_min) / (q_max - q_min + 1e-9)
    else:
        norm_qualities = np.zeros_like(qualities)

    # Build similarity kernel in feature space
    X_pool = np.array([cand["x"] for cand in context["pool"]])
    
    # Compute pairwise Euclidean distances between candidates (excluding self-similarity by setting diagonal to large value)
    dists = np.linalg.norm(X_pool[:, None] - X_pool, axis=2)  # shape: [n_candidates, n_candidates]
    np.fill_diagonal(dists, np.inf)

    # Use Gaussian kernel for similarity
    sigma = np.median(dists[dists < np.inf]) / 2.0 if not np.all(np.isinf(dists)) else 1.0
    
    similarities = np.exp(-dists ** 2 / (2 * sigma**2))
    
    # Set diagonal to zero so each candidate doesn't penalize itself
    np.fill_diagonal(similarities, 0)

    # Compute diversity scores as average similarity of a point with all others in the pool,
    # but we want high diversity = low similarity -> invert it by subtracting from max possible (1)
    diversities = []
    
    for i in range(len(context["pool"])):
        sim_to_others = similarities[i]
        
        if np.all(sim_to_others == 0):
            div_score = 1.0
        else:
            # Average similarity excluding self, then invert to get diversity score [0..1] 
            avg_sim_i = np.mean(sim_to_others)
            
            # Normalize so that max possible average sim gives min (zero) diversity and vice versa.
            if not np.isinf(avg_sim_i):
                div_score = 1. - avg_sim_i
            else:
                div_score = 0.

        diversities.append(div_score)

    scores = norm_qualities * np.array(diversities)
    
    return list(scores)