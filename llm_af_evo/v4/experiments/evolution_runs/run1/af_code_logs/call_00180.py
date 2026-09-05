def score_pool(context):
    """weighted sum of: novelty_stagnation(4.74), novelty_min(2.98), mu_min(1.91), novelty_mean(2.14)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.7425) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (2.9758) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (1.9058) * (min(gp[name]['mean'] for name in names))
        s = s + (2.1393) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
