def score_pool(context):
    """weighted sum of: novelty_min(2.97), mu_sum(1.49)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.9712) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (1.4946) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
