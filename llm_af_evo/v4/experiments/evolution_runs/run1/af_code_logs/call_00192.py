def score_pool(context):
    """weighted sum of: novelty_stagnation(2.91), sigma_sum_norm(4.87), novelty_min(4.22)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.9064) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (4.8690) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.2175) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        scores.append(s)
    return scores
