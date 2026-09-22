def score_pool(context):
    """weighted sum of: novelty_stagnation(4.06), novelty_mean(2.34), mu_min(1.58)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.0584) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (2.3418) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        s = s + (1.5841) * (min(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
