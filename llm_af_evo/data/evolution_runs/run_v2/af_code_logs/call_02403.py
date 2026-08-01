def score_pool(context):
    """weighted sum of: novelty_min(2.76), novelty_stagnation(4.03)"""
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.7638) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (4.0340) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        scores.append(s)
    return scores
