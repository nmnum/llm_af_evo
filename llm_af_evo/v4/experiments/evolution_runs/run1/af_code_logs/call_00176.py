def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(3.69), mu_min(1.91), sigma_max_norm(4.38), novelty_mean(4.38)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.6896) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (1.9085) * (min(gp[name]['mean'] for name in names))
        s = s + (4.3835) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.3777) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
