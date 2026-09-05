def score_pool(context):
    """Rank candidates by a combination of normalized predicted quality and joint diversity; higher scores indicate both strong predictions and distinctiveness from other top-scoring candidates."""
    names = context["objective_names"]
    
    # Compute raw qualities (sums of means)
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)
    
    # Normalize quality to [0, 1]
    min_q, max_q = np.min(qualities), np.max(qualities)
    if max_q - min_q < 1e-9:
        norm_qualities = np.ones(len(qualities))
    else:
        norm_qualities = (np.array(qualities) - min_q) / (max_q - min_q)

    # Compute diversity scores
    x_vals = [cand["x"] for cand in context["pool"]]
    
    def compute_similarity_matrix(x_list):
        n = len(x_list)
        sim_mat = np.zeros((n, n))
        
        for i in range(n):
            diff = (np.array(x_list[i]) - np.vstack([np.array(xi) for xi in x_list])) ** 2
            distances_squared = np.sum(diff, axis=1)
            
            # Gaussian similarity kernel with bandwidth chosen to make typical values around 0.5
            sigma_sq = max(1e-6, np.mean(distances_squared))
            sim_mat[i] = np.exp(-distances_squared / (2 * sigma_sq))

        return sim_mat
    
    S_x = compute_similarity_matrix(x_vals)
    
    # Set diagonal to zero so each candidate doesn't penalize itself
    np.fill_diagonal(S_x, 0)

    diversity_scores = []
    for i in range(len(context["pool"])):
        
        similarity_to_others = [S_x[i][j] if j != i else -1.0 
                                for j in range(len(context["pool"]))]

        # Use only the top candidates (e.g., those with highest normalized quality) to compute diversity
        threshold_idx = int(0.5 * len(similarity_to_others))  # consider top half as similar
    
        sorted_indices = np.argsort(-np.array(norm_qualities))
        
        relevant_similarities = [sim for idx, sim in enumerate(similarity_to_others)
                                if idx in set(sorted_indices[:threshold_idx])]
    
        diversity_score_i = (1.0 - max(relevant_similarities))  # higher is better
        diversity_scores.append(diversity_score_i)

    scores = norm_qualities * np.array(diversity_scores) 
    
    return list(scores)