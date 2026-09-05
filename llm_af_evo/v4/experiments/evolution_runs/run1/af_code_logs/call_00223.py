def score_pool(context):
    """weighted sum of: novelty_min(2.16), sigma_max_norm(2.15)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.1602) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (2.1456) * (max(gp[name]['std']/front_range[name] for name in names))
        scores.append(s)
    return scores
