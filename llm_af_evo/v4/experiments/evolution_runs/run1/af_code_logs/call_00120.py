def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(0.67), novelty_min(4.05), sigma_max_norm(3.81)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (0.6738) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (4.0512) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (3.8067) * (max(gp[name]['std']/front_range[name] for name in names))
        scores.append(s)
    return scores
