def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(3.66), sigma_sum_norm(1.38), mu_min(2.77), mu_sum(2.20), novelty_mean(3.32)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.6645) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (1.3776) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.7684) * (min(gp[name]['mean'] for name in names))
        s = s + (2.1956) * (sum(gp[name]['mean'] for name in names))
        s = s + (3.3218) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
