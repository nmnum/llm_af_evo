def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(3.34), sigma_sum_norm(0.79), novelty_min(4.23), novelty_mean(2.70)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.3375) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (0.7928) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.2264) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (2.7037) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
