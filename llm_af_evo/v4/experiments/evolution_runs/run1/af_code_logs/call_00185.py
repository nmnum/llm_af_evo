def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(2.81), novelty_stagnation(2.80), novelty_min(0.37), mu_min(0.58), mu_sum(3.13), novelty_mean(2.27)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.8135) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (2.8021) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (0.3718) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (0.5811) * (min(gp[name]['mean'] for name in names))
        s = s + (3.1303) * (sum(gp[name]['mean'] for name in names))
        s = s + (2.2717) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
