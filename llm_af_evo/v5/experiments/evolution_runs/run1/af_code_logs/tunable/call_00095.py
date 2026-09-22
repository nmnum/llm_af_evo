def score_pool(context):
    """Blend acquisition value with novelty based on distance to top candidates and Pareto front coverage gaps."""
    from scipy.spatial.distance import cdist
    
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Normalize acq_value_norm across the pool
    acqs = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    if len(acqs) > 1:
        a_min, a_max = acqs.min(), acqs.max()
        norm_acqs = (acqs - a_min) / max(a_max - a_min, 1e-9)
    else:
        norm_acqs = np.array([0.5])
    
    # Compute distances to Pareto front for coverage scoring
    use_pf = len(pf) >= 3  
    ref_points = pf if use_pf else context["Y_obs"]
        
    gap_scores = []
    for cand in context["pool"]:
        pred_obj = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        dists = cdist(pred_obj.reshape(1, -1), ref_points)[0]
        nearest_dists = sorted(dists)[:3 if use_pf else min(len(ref_points), 3)]
        coverage_gap_score = np.mean(nearest_dists) 
        gap_scores.append(coverage_gap_score)
    
    # Normalize gaps
    g_min, g_max = min(gap_scores), max(gap_scores)
    norm_gaps = (np.array(gap_scores) - g_min) / max(g_max - g_min, 1e-9)

    # Diversity: inverse similarity to top acq candidates  
    pool_size = len(context["pool"])
    
    x_vals = [cand['x'] for cand in context["pool"]]
    K_sim = np.zeros((pool_size, pool_size))
    for i in range(pool_size):
        dists_sq = [np.sum((x_vals[i] - x_vals[j]) ** 2) for j in range(pool_size)]
        dists_sq[i] = np.inf
        K_sim[:,i] = np.exp(-0.5 * np.array(dists_sq))
    
    top_k_candidates = min(3, pool_size)
    sorted_indices = list(np.argsort(norm_acqs)[::-1])[:top_k_candidates]
        
    avg_sims = []
    for j in range(pool_size):
        if len(sorted_indices) > 0:
            sim_to_top_k = sum(K_sim[j][i] for i in sorted_indices)
            avg_similarity = sim_to_top_k / float(len(sorted_indices))
        else:
            avg_similarity = np.random.rand()
        avg_sims.append(avg_similarity)

    diversity_scores = [1. - s for s in avg_sims]

    # Final score: blend acquisition, gap coverage and novelty
    scores = []
    alpha, beta, gamma = 0.65, 0.2, 0.15   # weights
    
    for i in range(pool_size):
        blended_score = (alpha * norm_acqs[i] + 
                         beta * (1 - norm_gaps[i]) +
                         gamma * diversity_scores[i])
        scores.append(blended_score)
        
    return scores