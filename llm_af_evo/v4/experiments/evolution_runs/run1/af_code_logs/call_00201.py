def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(0.93), mu_min(3.05), sigma_max_norm(1.97), mu_sum(2.50)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (0.9325) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (3.0498) * (min(gp[name]['mean'] for name in names))
        s = s + (1.9684) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.4988) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
