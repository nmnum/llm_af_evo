def score_pool(context):
    """weighted sum of: novelty_stagnation(4.02), novelty_min(0.40)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.0248) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (0.4001) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        scores.append(s)
    return scores
