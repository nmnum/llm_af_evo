def score_pool(context):
    """Rank candidates by normalized quality multiplied by diversity from high-scoring peers."""
    names = context["objective_names"]
    
    # Compute raw qualities (sum of predicted means)
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)
    
    # Normalize quality to [0, 1]
    min_q, max_q = np.min(qualities), np.max(qualities)
    if max_q - min_q < 1e-9:
        norm_qualities = np.ones_like(qualities)
    else:
        norm_qualities = (np.array(qualities) - min_q) / (max_q - min_q + 1e-9)

    # Build similarity kernel in feature space
    X_pool = np.stack([cand["x"] for cand in context["pool"]])
    
    # Pairwise squared Euclidean distances between candidates' features
    dists_sq = np.sum((X_pool[:, None] - X_pool[None, :]) ** 2, axis=2)
    
    # Convert to similarity (excluding self-similarity by setting diagonal to zero)
    similarities = np.exp(-dists_sq) 
    np.fill_diagonal(similarities, 0.0)

    scores = []
    for i in range(len(context["pool"])):
        quality_score = norm_qualities[i]
        
        # Compute diversity as average similarity with all other candidates
        avg_sim = similarities[i].mean()
        diversity_score = 1 - avg_sim
        
        final_score = quality_score * diversity_score 
        scores.append(final_score)
    
    return scores