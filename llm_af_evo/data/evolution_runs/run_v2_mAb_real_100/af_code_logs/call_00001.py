def score_pool(context):
    """Rank candidates by inverse distance to the nearest previously-observed point, purely based on novelty."""
    scores = []
    X_obs = context["X_obs"]
    for cand in context["pool"]:
        x = cand["x"]
        distances = np.linalg.norm(X_obs - x, axis=1)
        min_distance = np.min(distances)
        # Use inverse distance as score, with a small epsilon to avoid division by zero
        score = 1.0 / (min_distance + 1e-8)
        scores.append(score)
    return scores