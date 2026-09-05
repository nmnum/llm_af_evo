def score_pool(context):
    """Rank candidates by normalized predicted quality multiplied by joint diversity; diversity is based on exponential similarity kernel between candidates in feature space."""
    names = context["objective_names"]
    
    # Extract features and predictions for all candidates
    Xs = np.array([cand['x'] for cand in context['pool']])
    qualities = []
    for cand in context['pool']:
        gp_posterior = cand['gp_posterior']
        quality_sum = sum(gp_posterior[name]["mean"] for name in names)
        qualities.append(quality_sum)

    # Normalize quality to [0, 1]
    min_qual = np.min(qualities)
    max_qual = np.max(qualities)
    if abs(max_qual - min_qual) < 1e-9:
        normed_quals = np.ones(len(qualities))
    else:
        normed_quals = (np.array(qualities) - min_qual) / (max_qual - min_qual)

    # Compute similarity kernel in feature space
    n_cands = len(context['pool'])
    K_sim = np.zeros((n_cands, n_cands))

    for i in range(n_cands):
        diff_x = Xs[i] - Xs  # Shape: (n_cands,)
        dist_sq = np.sum(diff_x ** 2, axis=1) 
        sim_i = np.exp(-dist_sq)
        K_sim[i,:] = sim_i

    # Remove self-similarity
    np.fill_diagonal(K_sim, 0)

    # Compute diversity scores as inverse of average similarity to high-quality candidates (excluding itself).
    diversities = []
    
    for i in range(n_cands):
        if normed_quals[i] == 1.:
            div_score = 1.
        else: 
            similarities_to_high_qual = K_sim[i, :]
            # Avoid dividing by zero or including self
            avg_similarity = np.mean(similarities_to_high_qual[normed_quals > .5])
            if abs(avg_similarity) < 1e-9:
                div_score = 0.
            else: 
                div_score = max(0., (1. - avg_similarity)) # Inverse of similarity
        diversities.append(div_score)

    scores = normed_quals * np.array(diversities)
    
    return list(scores)