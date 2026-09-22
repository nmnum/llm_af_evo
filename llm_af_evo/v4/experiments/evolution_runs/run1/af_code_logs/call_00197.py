def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(2.30), sigma_sum_norm(1.10), sigma_max_norm(2.35), mu_sum(4.85)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.2965) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (1.0965) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.3502) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.8508) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
