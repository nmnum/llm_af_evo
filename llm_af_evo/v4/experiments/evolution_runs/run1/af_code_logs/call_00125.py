def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(2.23), novelty_mean(4.27), mu_min(3.29), mu_sum(1.44)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.2337) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (4.2741) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        s = s + (3.2860) * (min(gp[name]['mean'] for name in names))
        s = s + (1.4436) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
