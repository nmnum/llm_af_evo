def score_pool(context):
    """Rank candidates by normalized quality multiplied by diversity, where diversity is inverse similarity to other high-scoring candidates."""
    names = context["objective_names"]
    
    # Predicted qualities (sum of means) for each candidate
    qualities = np.array([sum(cand["gp_posterior"][name]["mean"] for name in names)
                          for cand in context["pool"]])
    
    # Normalize quality to [0, 1] range
    q_min, q_max = qualities.min(), qualities.max()
    if q_max > q_min:
        norm_qualities = (qualities - q_min) / (q_max - q_min + 1e-9)
    else:
        norm_qualities = np.zeros_like(qualities)

    # Compute pairwise similarities in feature space
    X_pool = np.array([cand["x"] for cand in context["pool"]])
    
    # Pairwise squared Euclidean distances between candidates (excluding self-similarity by setting diagonal to inf)
    dists_sq = np.sum((X_pool[:, None, :] - X_pool[None, :, :]) ** 2, axis=2) 
    np.fill_diagonal(dists_sq, np.inf)

    # Convert squared Euclidean distances into similarities (exp(-distance))
    sim_matrix = np.exp(-dists_sq)
    
    # Compute diversity scores as inverse of similarity to top candidates
    div_scores = []
    for i in range(len(context["pool"])):
        # Get indices and values of the highest quality neighbors, excluding self 
        neighbor_indices = np.argsort(norm_qualities)[::-1]
        
        # Only consider a reasonable number (e.g., 5) most similar high-quality candidates
        top_neighbors = [idx for idx in neighbor_indices if idx != i][:5]

        similarity_to_top = sim_matrix[i][top_neighbors].sum()
        diversity_score = np.exp(-similarity_to_top)
        
        div_scores.append(diversity_score)

    # Combine normalized quality and diversity scores (both positive, so multiplication is safe)  
    final_scores = norm_qualities * np.array(div_scores)
    
    return list(final_scores)