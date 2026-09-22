def score_pool(context):
    """weighted sum of: sigma_sum_norm(2.85), sigma_max_norm(4.85), mu_min(2.69)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.8543) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.8485) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.6942) * (min(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
