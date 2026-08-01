def score_pool(context):
    """Rank candidates by a diversity-adjusted quality score, preferring those that are both highly predicted and dissimilar to others in the pool."""
    names = context["objective_names"]
    scores = []
    
    # Build a similarity matrix based on predicted objective values
    n_candidates = len(context["pool"])
    similarity_matrix = np.zeros((n_candidates, n_candidates))
    
    for i, cand_i in enumerate(context["pool"]):
        for j, cand_j in enumerate(context["pool"]):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Use Euclidean distance in predicted objective space
                dist = np.sqrt(sum((cand_i["gp_posterior"][name]["mean"] - cand_j["gp_posterior"][name]["mean"])**2 for name in names))
                similarity_matrix[i, j] = np.exp(-dist)  # Convert distance to similarity
    
    # For each candidate, compute a score that balances quality and diversity
    for i, cand in enumerate(context["pool"]):
        # Use the mean of predicted objectives as quality measure
        quality = sum(cand["gp_posterior"][name]["mean"] for name in names)
        
        # Compute average similarity to all other candidates
        avg_similarity = np.mean(similarity_matrix[i])
        
        # Diversity score is inverse of average similarity (higher diversity = lower similarity)
        diversity = 1.0 - avg_similarity
        
        # Final score: quality adjusted by diversity
        scores.append(quality * diversity)
    
    return scores