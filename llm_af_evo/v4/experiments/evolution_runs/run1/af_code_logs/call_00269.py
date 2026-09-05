def score_pool(context):
    """Rank candidates by normalized quality multiplied by diversity; diversity is inverse similarity based on feature space distance."""
    names = context["objective_names"]
    
    # Compute raw qualities (sum of means)
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)
    
    # Normalize quality to [0, 1]
    min_qual = np.min(qualities)
    max_qual = np.max(qualities)
    if max_qual > min_qual:
        norm_qualities = [(q - min_qual) / (max_qual - min_qual + 1e-9) for q in qualities]
    else:
        norm_qualities = [0.0] * len(qualities)

    # Compute pairwise similarities based on feature space distance
    X_pool = np.array([cand["x"] for cand in context["pool"]])
    diff_matrix = X_pool[:, None, :] - X_pool[None, :, :]
    distances_squared = np.sum(diff_matrix**2, axis=2)
    
    # Convert to similarities (avoid self-similarity by setting diagonal to 0 after computation)  
    sim_matrix = np.exp(-distances_squared / (np.var(X_pool, axis=0).sum() + 1e-9))
    np.fill_diagonal(sim_matrix, 0)

    scores = []
    for i in range(len(context["pool"])):
        # Diversity is average similarity to all other candidates
        diversity_score = 1 - np.mean(sim_matrix[i])
        
        # Final score: normalized quality times diversity 
        final_score = norm_qualities[i] * diversity_score
        
        scores.append(final_score)
    
    return scores