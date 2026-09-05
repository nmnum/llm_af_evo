def score_pool(context):
    """Rank candidates by normalized quality multiplied by diversity; diversity is inverse similarity based on feature space distance."""
    names = context["objective_names"]
    pool = context["pool"]
    
    # Compute raw qualities (sum of predicted means)
    qualities = np.array([sum(cand["gp_posterior"][name]["mean"] for name in names) 
                          for cand in pool])
    
    # Normalize quality to [0, 1]
    q_min, q_max = qualities.min(), qualities.max()
    if q_max - q_min < 1e-9:
        norm_qualities = np.ones_like(qualities)
    else:
        norm_qualities = (qualities - q_min) / (q_max - q_min)
    
    # Compute pairwise similarity matrix in feature space
    X = np.array([cand["x"] for cand in pool])
    diff_matrix = X[:, None, :] - X[None, :, :]
    distances_sq = np.sum(diff_matrix**2, axis=2)
    similarities = np.exp(-distances_sq)  # Avoid self-similarity by design
    np.fill_diagonal(similarities, 0.0)

    # Compute diversity scores (inverse of average similarity to top candidates)
    n_top_candidates = max(1, len(pool)//4)  
    diversities = []
    
    for i in range(len(pool)):
        quality_ranking = norm_qualities.argsort()[::-1]
        top_indices = quality_ranking[:n_top_candidates] 
        similar_scores = similarities[i][top_indices].sum()
        
        # Normalize diversity to [0, 1], where higher means more diverse
        if len(top_indices) > 0:
            avg_similar_to_others = (similar_scores / len(top_indices))
            diversity_score = max(0.0, 1 - avg_similar_to_others)
        else:  
            diversity_score = 1
        
        diversities.append(diversity_score)

    # Combine normalized quality and diversity
    scores = norm_qualities * np.array(diversities) 
    
    return list(scores)