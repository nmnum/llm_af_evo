def score_pool(context):
    """Score candidates by normalized acquisition value adjusted for novelty and diversity."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Base acquisition values (already hypervolume improvement estimates)
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute distances from each candidate to the nearest previously observed point
    X_obs = context["X_obs"]
    if len(X_obs) == 0:
        novelty_penalties = np.zeros(pool_size)
    else:
        X_pool = np.array([cand['x'] for cand in context["pool"]])
        # Compute pairwise squared Euclidean distances between pool and observed points
        dists_sq = ((X_pool[:, None, :] - X_obs[None, :, :]) ** 2).sum(axis=2)
        min_dists_sq = np.min(dists_sq, axis=1)
        novelty_penalties = np.sqrt(min_dists_sq) / np.sqrt(4.0 * len(names)) # Normalized by feature dimensions
        
    # Combine base score with a penalty for proximity to existing points
    final_scores = base_scores - 0.5 * novelty_penalties
    
    return list(final_scores)