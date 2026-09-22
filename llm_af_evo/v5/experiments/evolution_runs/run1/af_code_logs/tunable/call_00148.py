def score_pool(context):
    """Score candidates by normalized quality multiplied by inverse diversity; quality is sum of means min-max normalised to [0,1], and diversity is 1 minus average similarity (exp(-distance)) with other high-scoring candidates."""
    names = context["objective_names"]
    
    # Compute raw qualities as sums of predicted means
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)
        
    min_q, max_q = np.min(qualities), np.max(qualities)
    
    # Normalize quality to [0, 1]
    if abs(max_q - min_q) < 1e-9:
        norm_qualities = np.array([0.5] * len(qualities))
    else:
        norm_qualities = (np.array(qualities) - min_q) / (max_q - min_q)
    
    # Compute pairwise similarities in feature space
    X_pool = np.stack([cand["x"] for cand in context["pool"]])
    distances_x = np.sum((X_pool[:, None, :] - X_pool[None, :, :]) ** 2, axis=2) ** 0.5  
    sim_matrix_x = np.exp(-distances_x)
    
    # Exclude self-similarity
    for i in range(len(sim_matrix_x)):
        sim_matrix_x[i,i] = 0
    
    scores = []
    threshold_q = np.percentile(norm_qualities, 75) 
    high_quality_indices = np.where(norm_qualities >= threshold_q)[0]
    
    # For each candidate compute inverse diversity
    for i in range(len(context["pool"])):
        if len(high_quality_indices) > 1:
            avg_sim_to_highq = sim_matrix_x[i, :][high_quality_indices].mean()
        else: 
            avg_sim_to_highq = np.mean(sim_matrix_x[i,:]) 
            
        diversity_score = max(0.0, 1 - avg_sim_to_highq)
        
        # Final score is normalized quality times inverse diversity
        final_score = norm_qualities[i] * diversity_score  
        scores.append(final_score)

    return scores