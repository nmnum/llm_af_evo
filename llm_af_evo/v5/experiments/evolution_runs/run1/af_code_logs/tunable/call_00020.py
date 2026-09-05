def score_pool(context):
    """Combine acquisition value with a novelty penalty that suppresses nearby duplicates, favoring diverse exploration."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Base scores from qLogNEHVI (already normalized)
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute pairwise distances between candidates
    X_pool = np.stack([cand["x"] for cand in context["pool"]])  # shape: [n, d]
    dists = np.sum((X_pool[:, None] - X_pool[None])**2, axis=-1)**0.5
    
    # Suppress nearby duplicates by reducing score of candidates within a threshold
    novelty_threshold = 0.1 * (np.max(dists) if np.max(dists) > 0 else 1)
    
    final_scores = base_scores.copy()
    for i in range(pool_size):
        near_indices = np.where((dists[i] < novelty_threshold) & (distances[i] != 0))[0]
        # Reduce score of nearby candidates
        if len(near_indices) > 0:
            penalty_factor = min(1.0, float(len(near_indices)) / max(5., pool_size * .2))
            final_scores[i] *= (1 - penalty_factor)
    
    return list(final_scores)