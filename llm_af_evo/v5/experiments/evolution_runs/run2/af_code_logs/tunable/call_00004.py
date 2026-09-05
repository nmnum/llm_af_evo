def score_pool(context):
    """Rank candidates by quality-diversity tradeoff using joint similarity kernel."""
    names = context["objective_names"]
    
    # Compute raw qualities (sum of predicted means)
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)
    
    # Normalize quality to [0, 1] range
    min_q, max_q = np.min(qualities), np.max(qualities)
    if abs(max_q - min_q) < 1e-9:
        norm_qualities = np.ones_like(qualities)
    else:
        norm_qualities = (np.array(qualities) - min_q) / (max_q - min_q + 1e-9)

    # Build similarity matrix in feature space
    X = np.stack([cand["x"] for cand in context["pool"]])
    distances_sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    
    # Compute Gaussian kernel (similarity) with bandwidth chosen to make median distance ~1
    sigma_sq = np.median(distances_sq.flatten()) + 1e-9  
    similarities = np.exp(-distances_sq / sigma_sq)

    # Set diagonal to zero so each candidate doesn't penalize itself
    np.fill_diagonal(similarities, 0.0)
    
    scores = []
    for i in range(len(context["pool"])):
        quality_score = norm_qualities[i]
        
        # Compute average similarity with top candidates (excluding self) 
        sim_scores = similarities[i]  
        diversity_penalty = np.mean(sim_scores)

        final_score = quality_score * (1.0 - diversity_penalty)
        scores.append(final_score)
    
    return scores