def score_pool(context):
    """Score by acquisition value modulated by how much a candidate fills gaps in objective space coverage, with repulsion from high-scoring neighbors."""
    names = context["objective_names"]
    
    # Compute raw qualities (sum of means)
    qualities = []
    for cand in context["pool"]:
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
        qualities.append(mu_sum)

    q_min, q_max = min(qualities), max(qualities)
    
    # Normalize quality to [0, 1]
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.array([0.5] * len(qualities))
    else:
        norm_qualities = (np.array(qualities) - q_min) / (q_max - q_min)
    
    # Get acquisition values
    acqs = [cand["acq_value_norm"] for cand in context["pool"]]
        
    scores = []
    top_k_candidates = min(len(context['pool']), 5)

    sorted_indices_by_quality = list(np.argsort(norm_qualities)[::-1])[:top_k_candidates]
    
    # Repulsion based on similarity to high-quality candidates
    x_vals = [cand['x'] for cand in context["pool"]]
        
    K_sim = np.zeros((len(context["pool"]), len(context["pool"])))
    for i in range(len(context["pool"])):
        dists_sq = [np.sum((x_vals[i] - x_vals[j]) ** 2) for j in range(len(context["pool"])) if j !=i]
        K_sim[:,i] = np.exp(-0.5 * np.array(dists_sq + [1e9])) # Set self-similarity to zero
    
    repulsion_scores = []
    
    for i, cand_idx in enumerate(range(len(context['pool']))):
        
        sim_to_top_quality =  sum(K_sim[cand_idx][j] 
                                  for j in sorted_indices_by_quality if j != cand_idx)
                
        num_comparisons = len(sorted_indices_by_quality) - (1 if cand_idx in sorted_indices_by_quality else 0)

        avg_similarity =sim_to_top_quality / max(1., float(num_comparisons))
        
        # Apply repulsion: lower score for candidates similar to top ones
        repulsion_factor = np.exp(-avg_similarity * 2.5)
            
        final_score_i = acqs[i] + (0.3) * norm_qualities[i]*repulsion_factor
        
        scores.append(final_score_i)

    return scores