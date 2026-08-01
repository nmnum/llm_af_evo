def score_pool(context):
    """Score candidates by the distance to their nearest observed neighbor in objective space, rewarding novelty."""
    scores = []
    Y_obs = context["Y_obs"]
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
        distances = np.linalg.norm(Y_obs - gp_mean, axis=1)
        nearest_distance = np.min(distances)
        scores.append(nearest_distance)
    return scores