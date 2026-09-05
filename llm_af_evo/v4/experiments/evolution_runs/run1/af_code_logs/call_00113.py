def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(2.09), novelty_stagnation(1.23), novelty_mean(1.40), sigma_max_norm(1.60)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.0924) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (1.2337) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (1.3970) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        s = s + (1.6019) * (max(gp[name]['std']/front_range[name] for name in names))
        scores.append(s)
    return scores
