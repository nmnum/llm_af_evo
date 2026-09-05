def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(4.29), novelty_stagnation(4.03), novelty_min(0.71), sigma_max_norm(2.27), sigma_sum_norm(0.95)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.2870) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (4.0310) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (0.7097) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (2.2671) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (0.9518) * (sum(gp[name]['std']/front_range[name] for name in names))
        scores.append(s)
    return scores
