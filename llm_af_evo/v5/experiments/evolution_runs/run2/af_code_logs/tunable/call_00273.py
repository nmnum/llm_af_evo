def score_pool(context):
    """Rank candidates by normalized quality multiplied by inverse diversity; diversity is based on x-space similarity kernel."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Compute raw qualities (sum of means)
    qualities = np.array([sum(cand['gp_posterior'][name]['mean'] for name in names) 
                          for cand in context["pool"]])
    
    # Normalize quality to [0, 1]
    q_min, q_max = qualities.min(), qualities.max()
    if q_max - q_min < 1e-9:
        norm_qualities = np.zeros_like(qualities)
    else:
        norm_qualities = (qualities - q_min) / (q_max - q_min + 1e-9)

    # Compute similarity kernel in x-space
    X_pool = np.array([cand['x'] for cand in context["pool"]])
    diff_matrix = X_pool[:, None, :] - X_pool[None, :, :]
    distances_sq = np.sum(diff_matrix**2, axis=2)
    
    # Avoid division by zero and set diagonal to 0
    kernel = np.exp(-distances_sq) 
    np.fill_diagonal(kernel, 0)

    scores = []
    for i in range(pool_size):
        quality_score = norm_qualities[i]
        
        # Compute inverse diversity (1 - avg similarity)
        similarities = kernel[i]  
        if not np.any(similarities > 0):   # No similar candidates
            inv_diversity = 1.0 
        else:
            mean_similarity = np.mean(similarities)   
            inv_diversity = max(0., 1 - mean_similarity)
        
        score = quality_score * inv_diversity  
        scores.append(score)

    return scores