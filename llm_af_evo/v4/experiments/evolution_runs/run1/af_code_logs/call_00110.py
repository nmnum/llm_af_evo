def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(0.80), novelty_mean(2.91), sigma_max_norm(0.17), novelty_min(3.83)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (0.8007) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (2.9146) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        s = s + (0.1673) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (3.8348) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        scores.append(s)
    return scores
