def score_pool(context):
    """weighted sum of: novelty_stagnation(4.11), sigma_sum_norm(3.30), novelty_min(0.42), sigma_max_norm(3.54)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.1079) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (3.2980) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (0.4245) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (3.5386) * (max(gp[name]['std']/front_range[name] for name in names))
        scores.append(s)
    return scores
