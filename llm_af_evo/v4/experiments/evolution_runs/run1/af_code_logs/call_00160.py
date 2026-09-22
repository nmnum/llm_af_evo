def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(3.22), sigma_sum_norm(0.91), sigma_max_norm(0.18), novelty_min(0.50)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.2183) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (0.9084) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (0.1817) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (0.5028) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        scores.append(s)
    return scores
