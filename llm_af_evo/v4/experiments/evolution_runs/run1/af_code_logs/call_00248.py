def score_pool(context):
    """weighted sum of: sigma_max_norm(3.59), sigma_sum_norm(2.52), mu_sum(1.29), sigma_sum_norm_early(0.76), mu_min(4.09)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.5930) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.5185) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (1.2884) * (sum(gp[name]['mean'] for name in names))
        s = s + (0.7643) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (4.0933) * (min(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
