def score_pool(context):
    """weighted sum of: novelty_stagnation(0.82), novelty_min(3.46), mu_min(1.15), novelty_mean(2.44)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (0.8197) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (3.4624) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (1.1521) * (min(gp[name]['mean'] for name in names))
        s = s + (2.4427) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
