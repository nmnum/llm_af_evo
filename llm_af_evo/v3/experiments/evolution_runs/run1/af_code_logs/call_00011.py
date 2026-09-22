def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(0.11), sigma_sum_norm(4.30)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (0.1134) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (4.3013) * (sum(gp[name]['std']/front_range[name] for name in names))
        scores.append(s)
    return scores
