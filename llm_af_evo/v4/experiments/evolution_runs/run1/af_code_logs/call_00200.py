def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(0.86), novelty_stagnation(0.94), novelty_min(4.59), mu_min(1.13), mu_sum(0.51)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (0.8572) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (0.9428) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (4.5857) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (1.1344) * (min(gp[name]['mean'] for name in names))
        s = s + (0.5122) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
