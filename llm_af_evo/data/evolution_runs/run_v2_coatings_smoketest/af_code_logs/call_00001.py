def score_pool(context):
    """Score purely by novelty: rank candidates by distance to the nearest previously-observed point, ignoring predicted objective values entirely."""
    scores = []
    X_obs = context["X_obs"]
    for cand in context["pool"]:
        x = cand["x"]
        # Compute distances to all observed points
        distances = np.linalg.norm(X_obs - x, axis=1)
        # Novelty score is inverse of the nearest distance (higher = more novel)
        nearest_distance = np.min(distances)
        if nearest_distance == 0:
            # If candidate exactly matches an observed point, set score to very low value
            scores.append(-1e10)
        else:
            scores.append(1.0 / nearest_distance)
    return scores