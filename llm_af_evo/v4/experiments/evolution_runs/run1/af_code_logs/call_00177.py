def score_pool(context):
    """weighted sum of: sigma_sum_norm(3.37), mu_sum(4.08)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.3663) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.0828) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
