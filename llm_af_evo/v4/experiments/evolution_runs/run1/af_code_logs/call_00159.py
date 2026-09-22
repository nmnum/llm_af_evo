def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(3.01), sigma_sum_norm(4.91), mu_sum(3.47)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.0115) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (4.9101) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (3.4657) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
