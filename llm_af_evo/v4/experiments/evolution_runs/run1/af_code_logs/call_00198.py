def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(4.55), novelty_stagnation(2.93), sigma_max_norm(1.68), mu_sum(0.32)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.5507) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (2.9267) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (1.6754) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (0.3153) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
