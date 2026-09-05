def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(0.91), novelty_min(1.98), sigma_sum_norm(0.16)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (0.9078) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (1.9783) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (0.1637) * (sum(gp[name]['std']/front_range[name] for name in names))
        scores.append(s)
    return scores
