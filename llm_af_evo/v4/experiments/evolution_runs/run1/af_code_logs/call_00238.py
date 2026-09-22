def score_pool(context):
    """weighted sum of: novelty_mean(3.43), mu_min(1.15)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.4334) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        s = s + (1.1540) * (min(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
