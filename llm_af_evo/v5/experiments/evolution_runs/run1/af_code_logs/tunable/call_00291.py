def score_pool(context):
    """Score candidates by normalized quality multiplied by diversity; diversity is inverse similarity to top-ranked candidates."""
    names = context["objective_names"]
    
    # Compute raw qualities (sum of means)
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)
        
    q_min, q_max = min(qualities), max(qualities)
    
    # Normalize quality to [0, 1]
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.array([0.558] * len(qualities))
    else:
        norm_qualities = (np.array(qualities) - q_min) / (q_max - q_min)
    
    # Compute diversity scores based on similarity to top candidates
    pool_size = len(context["pool"])
    x_vals = [cand['x'] for cand in context["pool"]]
    
    # Build kernel matrix: exp(-||xi-xj||^2), excluding diagonal (self-similarity)  
    K_sim = np.zeros((pool_size, pool_size))
    for i in range(pool_size):
        dists_sq = [np.sum((x_vals[i] - x_vals[j]) ** 2) for j in range(pool_size)]
        # Set self-distance to large value so similarity is near zero
        dists_sq[i] = np.inf  
        K_sim[:,i] = np.exp(-0.5842 * np.array(dists_sq))
    
    diversity_scores = []
    top_k_candidates = min(3, pool_size)  # Top candidates for comparison
    
    sorted_indices = list(np.argsort(norm_qualities)[::-1])[:top_k_candidates]
    
    if not sorted_indices:
        return [norm_qualities[i] * (0.5805 + np.random.rand() / 2.) 
                for i in range(pool_size)]
        
    # For each candidate, compute average similarity to top candidates
    avg_similarities = []
    for j in range(pool_size):
        sim_to_top_k = sum(K_sim[j][i] for i in sorted_indices)
        if len(sorted_indices) > 0:
            avg_similarity = sim_to_top_k / float(len(sorted_indices))
        else:
            # fallback to random score
            avg_similarity = np.random.rand() 
        avg_similarities.append(avg_similarity)

    diversity_scores = [1. - s for s in avg_similarities]
    
    scores = []
    for i, (q_norm, d_score) in enumerate(zip(norm_qualities, diversity_scores)):
        # Multiply normalized quality with inverse similarity to top candidates
        score_i = q_norm * max(0., min(d_score, 1.)) 
        scores.append(score_i)
        
    return scores