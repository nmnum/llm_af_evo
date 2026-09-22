def score_pool(context):
    """Score candidates by acquisition value enhanced with novelty based on distance to observed points and diversity from top-ranked pool members."""
    names = context["objective_names"]
    
    # Use precomputed acq_value_norm directly for quality signal  
    qualities = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Normalize qualities
    q_min, q_max = qualities.min(), qualities.max()
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.full_like(qualities, 0.5)
    else: 
        norm_qualities = (qualities - q_min) / (q_max - q_min)

    # Compute novelty based on distance to previously observed points
    y_obs = context["Y_obs"]
    novelties = []
    
    for cand in context["pool"]:
        mu = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        distances = np.linalg.norm(y_obs - mu, axis=1)
        nearest_distance = np.min(distances) 
        # Invert so larger distance means higher novelty
        novelties.append(1.0 / (1e-9 + nearest_distance))
    
    norm_novelties = np.array(novelties)
    n_min, n_max = norm_novelties.min(), norm_novelties.max()
    if abs(n_max - n_min) < 1e-9:
        norm_novelties = np.full_like(norm_novelties, 0.5)
    else:
        norm_novelties = (norm_novelties - n_min) / (n_max - n_min)

    # Compute diversity from top candidates using feature space similarity
    pool_size = len(context["pool"])
    x_vals = np.array([cand['x'] for cand in context["pool"]])
    
    K_sim = np.zeros((pool_size, pool_size))
    for i in range(pool_size):
        dists_sq = [np.sum((x_vals[i] - x_vals[j]) ** 2) for j in range(pool_size)]
        # Set self-distance to large value so similarity is near zero
        dists_sq[i] = np.inf  
        K_sim[:,i] = np.exp(-0.5 * np.array(dists_sq))
    
    top_k_candidates = min(3, pool_size)
    sorted_indices = list(np.argsort(norm_qualities)[::-1])[:top_k_candidates]
        
    avg_similarities = []
    for j in range(pool_size):
        if len(sorted_indices) > 0:
            sim_to_top_k = sum(K_sim[j][i] for i in sorted_indices)
            avg_similarity = sim_to_top_k / float(len(sorted_indices))
        else: 
            # fallback to random similarity
            avg_similarity = np.random.rand()
            
        avg_similarities.append(avg_similarity)

    diversity_scores = [1. - s for s in avg_similarities]
    
    scores = []
    alpha, beta = 0.75, 0.25   # Weight balance between quality and novelty/diversity
    
    for q_norm, n_score, d_score in zip(norm_qualities, norm_novelties, diversity_scores):
        score_i = (alpha * q_norm + 
                   beta * n_score +
                  (1 - alpha - beta) * max(0., min(d_score, 1.)))
        
        scores.append(score_i)
    
    return scores