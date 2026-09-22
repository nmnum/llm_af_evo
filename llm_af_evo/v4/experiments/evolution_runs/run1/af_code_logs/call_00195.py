def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(0.64), novelty_stagnation(3.23), sigma_sum_norm(2.40), mu_sum(1.32)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (0.6440) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (3.2297) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (2.4038) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (1.3172) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
