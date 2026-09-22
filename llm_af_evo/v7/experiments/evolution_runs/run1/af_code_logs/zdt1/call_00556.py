def modifier(context):
    """Suppress candidates that are close in feature space to already-observed points, with a strength decaying as campaign progresses."""
    if len(context["X_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    values = []
    progress_factor = 1.0 - context["campaign"]["progress"]
    penalty_strength = 0.5 * progress_factor
    
    for i, cand in enumerate(context["pool"]):
        cand_x = cand["x"]
        
        # Compute squared distances to all previously observed points
        dists_sq = np.sum((context["X_obs"] - cand_x) ** 2, axis=1)
        min_dist_sq = np.min(dists_sq)

        # Apply suppression penalty if candidate is too close (within a threshold in x-space)
        threshold_sq = 0.05**2
        if min_dist_sq < threshold_sq:
            suppress_factor = np.exp(-min_dist_sq / threshold_sq) * penalty_strength
            values.append(-suppress_factor)
        else:
            values.append(0.0)

    return values