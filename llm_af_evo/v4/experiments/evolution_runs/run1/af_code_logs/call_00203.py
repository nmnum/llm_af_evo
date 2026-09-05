def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(1.93), novelty_stagnation(0.97), sigma_sum_norm(0.71)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (1.9272) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (0.9742) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (0.7121) * (sum(gp[name]['std']/front_range[name] for name in names))
        scores.append(s)
    return scores
