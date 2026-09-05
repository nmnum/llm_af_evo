def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(0.34), novelty_stagnation(3.32), novelty_min(1.22), sigma_sum_norm(3.96), mu_sum(2.91)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (0.3353) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (3.3161) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (1.2220) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (3.9585) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.9083) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
