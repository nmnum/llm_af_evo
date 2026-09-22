def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(1.34), novelty_mean(2.15), mu_sum(4.23)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (1.3389) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (2.1483) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        s = s + (4.2252) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
