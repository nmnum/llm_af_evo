def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(4.65), sigma_sum_norm(4.67), sigma_max_norm(1.26), mu_sum(0.20)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.6541) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (4.6691) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (1.2580) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (0.1989) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
