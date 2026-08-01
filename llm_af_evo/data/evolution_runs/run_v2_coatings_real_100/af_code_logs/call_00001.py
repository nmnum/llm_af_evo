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
        scores.append(1.0 / (nearest_distance + 1e-12))  # Add small epsilon to avoid division by zero
    return scores