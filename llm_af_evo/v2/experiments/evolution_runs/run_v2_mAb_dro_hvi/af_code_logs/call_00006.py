def score_pool(context):
    """Rank candidates by normalized quality multiplied by diversity, where diversity is inverse similarity to other high-scoring candidates."""
    names = context["objective_names"]
    
    # Predicted qualities (sum of means)
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)
    qualities = np.array(qualities)
    
    # Normalize quality to [0, 1]
    q_min, q_max = qualities.min(), qualities.max()
    if q_max - q_min < 1e-9:
        norm_qualities = np.zeros_like(qualities)
    else:
        norm_qualities = (qualities - q_min) / (q_max - q_min + 1e-9)

    # Compute pairwise similarities in feature space
    X = np.array([cand["x"] for cand in context["pool"]])
    
    # Pairwise squared Euclidean distances
    dists_sq = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
    
    # Similarity as exp(-distance), avoiding self-similarity by setting diagonal to zero
    similarities = np.exp(-dists_sq)
    np.fill_diagonal(similarities, 0)

    scores = []
    for i in range(len(context["pool"])):
        quality_score = norm_qualities[i]
        
        # Diversity: average similarity with all other candidates (excluding self)
        avg_sim = similarities[i].mean()
        diversity_score = 1 - avg_sim
        
        score = quality_score * diversity_score
        scores.append(score)

    return scores