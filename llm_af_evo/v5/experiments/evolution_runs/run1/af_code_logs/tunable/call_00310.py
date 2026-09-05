def score_pool(context):
    """Score candidates by normalized acquisition value multiplied by inverse diversity from top-ranked candidates."""
    names = context["objective_names"]
    
    # Use precomputed acq_value_norm directly as quality measure
    qualities = np.array([cand['acq_value_norm'] for cand in context["pool"]])
        
    q_min, q_max = qualities.min(), qualities.max()
    
    # Normalize quality to [0, 1]
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.array([0.5] * len(qualities))
    else:
        norm_qualities = (qualities - q_min) / (q_max - q_min)
    
    # Compute diversity scores based on similarity to top candidates
    pool_size = len(context["pool"])
    x_vals = [cand['x'] for cand in context["pool"]]
    
    # Build kernel matrix: exp(-||xi-xj||^2), excluding diagonal (self-similarity)  
    K_sim = np.zeros((pool_size, pool_size))
    for i in range(pool_size):
        dists_sq = [np.sum((x_vals[i] - x_vals[j]) ** 2) for j in range(pool_size)]
        # Set self-distance to large value so similarity is near zero
        dists_sq[i] = np.inf  
        K_sim[:,i] = np.exp(-0.5 * np.array(distsSq))
    
    diversity_scores = []
    top_k_candidates = min(3, pool_size)  # Top candidates for comparison
    
    sorted_indices = list(np.argsort(norm_qualities)[::-1])[:top_k_candidates]
    
    if not sorted_indices:
        return [norm_qualities[i] * (0.5 + np.random.rand() / 2.) 
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