def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(3.27), novelty_stagnation(1.00), sigma_sum_norm(2.87), sigma_max_norm(2.14), mu_sum(4.94)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.2655) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (0.9951) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (2.8722) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.1356) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.9396) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
