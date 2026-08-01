def score_pool(context):
    """Rank candidates by normalized predicted quality multiplied by diversity from high-scoring peers."""
    names = context["objective_names"]
    pool = context["pool"]
    
    # Compute raw quality as sum of predicted means
    qualities = np.array([sum(cand["gp_posterior"][name]["mean"] for name in names) for cand in pool])
    
    # Normalize quality to [0, 1] range
    q_min, q_max = qualities.min(), qualities.max()
    if q_max > q_min:
        norm_qualities = (qualities - q_min) / (q_max - q_min + 1e-9)
    else:
        norm_qualities = np.zeros_like(qualities)
    
    # Build similarity kernel based on x-space distances
    X = np.array([cand["x"] for cand in pool])
    distances = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(distances, np.inf)  # Exclude self-similarity
    similarity = np.exp(-distances)  # Similarity matrix
    
    # For each candidate, compute diversity as mean similarity to top-10% highest quality candidates
    n_top = max(1, len(pool) // 10)
    top_indices = np.argsort(norm_qualities)[-n_top:]
    
    diversities = []
    for i in range(len(pool)):
        sim_to_top = similarity[i, top_indices]
        diversity = np.mean(sim_to_top)
        diversities.append(diversity)
    
    # Combine quality and diversity
    scores = norm_qualities * (1 - np.array(diversities))
    return scores.tolist()