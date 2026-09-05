def score_pool(context):
    """Score candidates by their nearest-neighbor distance in objective space to all previously observed points."""
    scores = []
    y_obs = context["Y_obs"]
    for cand in context["pool"]:
        mu = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
        distances = np.linalg.norm(y_obs - mu, axis=1)
        nearest_distance = np.min(distances)
        scores.append(nearest_distance)
    return scores