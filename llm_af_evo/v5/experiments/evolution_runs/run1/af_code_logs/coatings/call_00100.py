def score_pool(context):
    """Score candidates by normalized acquisition value multiplied by inverse diversity from top-ranked batch."""
    names = context["objective_names"]
    
    # Use precomputed acq_value_norm directly as quality metric 
    qualities = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    q_min, q_max = qualities.min(), qualities.max()
    
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.full_like(qualities, 0.5)
    else:
        norm_qualities = (qualities - q_min) / (q_max - q_min)

    # Compute diversity based on feature space similarity to top candidates
    pool_size = len(context["pool"])
    
    if pool_size < 2: 
        return list(norm_qualities)
        
    x_vals = np.array([cand['x'] for cand in context["pool"]])
    
    K_sim = np.zeros((pool_size, pool_size))
    # Compute squared Euclidean distances
    dists_sq = np.sum((x_vals[:, None] - x_vals[None,:]) ** 2, axis=-1)
        
    # Set diagonal to large value so self-similarity is near zero 
    np.fill_diagonal(dists_sq, np.inf) 
    
    K_sim = np.exp(-0.5 * dists_sq)

    top_k_candidates = min(3, pool_size)
    
    sorted_indices = list(np.argsort(norm_qualities)[::-1][:top_k_candidates])
        
    if not sorted_indices:
        # fallback to random score
        return [norm_qualities[i] * (0.5 + np.random.rand() / 2.) 
                for i in range(pool_size)]
            
    
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