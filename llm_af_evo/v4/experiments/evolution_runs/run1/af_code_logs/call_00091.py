def score_pool(context):
    """Rank candidates by joint diversity-aware quality: normalize predicted objective sum to non-negative values, then multiply by inverse similarity (diversity) scores across all pool members."""
    names = context["objective_names"]
    
    # Extract predictions and compute raw qualities
    xs = np.array([cand['x'] for cand in context['pool']])
    means_list = [[cand['gp_posterior'][name]['mean'] for name in names] for cand in context['pool']]
    qualities = [sum(means) for means in means_list]
    
    # Normalize quality to non-negative range
    q_min, q_max = min(qualities), max(qualities)
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.array([0.5] * len(qualities))
    else:
        norm_qualities = (np.array(qualities) - q_min) / (q_max - q_min + 1e-9)
    
    # Compute similarity matrix in feature space
    n_cand = len(context['pool'])
    similarities = np.zeros((n_cand, n_cand))
    for i in range(n_cand):
        dists = np.linalg.norm(xs[i] - xs, axis=1)  # distances to all candidates including self (distance zero)
        exp_dists = np.exp(-dists ** 2 / (np.std(dists) ** 2 + 1e-9)) 
        similarities[i,:] = exp_dists
    
    # Exclude each candidate's similarity to itself by setting diagonal element
    np.fill_diagonal(similarities, 0)
    
    diversity_scores = []
    for i in range(n_cand):
        sim_sum = np.sum(similarities[:,i])  
        if abs(sim_sum) < 1e-9:
            # No similar candidates; full diversity score (avoid division by zero or near-zero sum of similarities)
            div_score = 1.0
        else: 
            inv_sim_avg = 1 / sim_sum   # higher inverse average similarity means lower overall similarity, i.e., more diverse.
            if np.isinf(inv_sim_avg):
                div_score = 1e9  
            elif not (np.isnan(div_score) or abs(div_score) > 1e6): 
                 inv_sim_avg = min(20.0, max(-20.0,inv_sim_avg)) # Clamp for stability
                 div_score = np.tanh(inv_sim_avg)
        diversity_scores.append(max(0.,div_score))
    
    scores = norm_qualities * diversity_scores
    
    return list(scores)