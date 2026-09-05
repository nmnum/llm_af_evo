def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(3.06), sigma_max_norm(2.62), mu_min(4.86)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.0590) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (2.6243) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.8610) * (min(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
