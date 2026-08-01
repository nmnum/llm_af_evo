def score_pool(context):
    """Pure novelty: rank by distance to the nearest observed point."""
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        scores.append(float(dists.min()))
    return scores