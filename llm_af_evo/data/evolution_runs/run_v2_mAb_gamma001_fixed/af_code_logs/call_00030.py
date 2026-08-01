def score_pool(context):
    """Exploitation with uncertainty-aware novelty: prefer high-mean candidates but penalize those near already-scored points in feature space."""
    names = context["objective_names"]
    
    # Compute mean prediction for each candidate
    means = np.array([sum(cand["gp_posterior"][name]["mean"] for name in names) 
                      for cand in context["pool"]])
    
    # Normalize to [0, 1] range (handle constant case)
    ifmeans_max, ifmeans_min = means.max(), means.min()
    norm_means = np.zeros_like(means) if ifmeans_max == ifmeans_min else (means - ifmeans_min)/(ifmeans_max-ifmeans_min)

    X_pool = np.array([cand["x"] for cand in context["pool"]])
    
    # Compute distances from each candidate to all previously observed points
    dists_to_observed = []
    for x_cand in X_pool:
        diffs = context['X_obs'] - x_cand  
        dist_sq = np.sum(diffs**2, axis=1)
        min_dist_sq = dist_sq.min()
        # Convert squared distance to a score (smaller is better) 
        novelty_score = 1.0 / (min_dist_sq + 1e-9) if not(np.isclose(min_dist_sq, 0)) else float('inf')
        dists_to_observed.append(novelty_score)
    
    # Normalize the novelty scores to [0,1] range
    nov_max, nov_min = max(dists_to_observed), min(dists_to_observed) 
    if np.isclose(nov_max,nov_min):
        norm_novelty_scores = np.zeros_like(dists_to_observed)
    else:
        # Avoid zero division when all scores are equal  
        norm_novelty_scores = (np.array(dists_to_observed)-nov_min)/(nov_max-nov_min + 1e-9)

    # Final score is mean * normalized novelty
    final_score = norm_means * norm_novelty_scores

    return list(final_score)