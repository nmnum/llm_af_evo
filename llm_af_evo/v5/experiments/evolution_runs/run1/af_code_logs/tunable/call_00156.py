def score_pool(context):
    """Rank candidates by normalized quality multiplied by inverse diversity; similarity computed in feature space."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Compute raw qualities (sum of means)
    qualities = np.array([sum(cand["gp_posterior"][name]["mean"] for name in names) 
                          for cand in context["pool"]])
    
    # Normalize quality to [0, 1]
    q_min, q_max = qualities.min(), qualities.max()
    if q_max > q_min:
        norm_qualities = (qualities - q_min) / (q_max - q_min + 1e-9)
    else:
        norm_qualities = np.zeros_like(qualities)

    # Compute feature similarity matrix
    X_features = np.array([cand["x"] for cand in context["pool"]])
    
    # Pairwise squared Euclidean distances between features (avoid self-similarity by setting diagonal to inf)
    dist_matrix_sq = -2 * X_features @ X_features.T + (
        np.sum(X_features**2, axis=1)[:, None] +
        np.sum(X_features**2, axis=1)[None, :]
    )
    
    # Set diagonals to large value (zero similarity for self-similarity)
    dist_matrix_sq[np.diag_indices_from(dist_matrix_sq)] = float('inf')
    
    similarities = np.exp(-dist_matrix_sq)  # Kernel: exp(-distance^2)

    # Compute diversity scores as mean dissimilarity from all other candidates
    diversity_scores = []
    for i in range(pool_size):
        sim_sum = np.sum(similarities[i, :]) 
        div_score = (1.0 - sim_sum / pool_size)  # Inverse similarity normalized to [0,1]
        diversity_scores.append(div_score)
    
    diversity_array = np.array(diversity_scores)

    scores = norm_qualities * diversity_array
    return list(scores)