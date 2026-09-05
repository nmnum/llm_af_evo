def score_pool(context):
    """weighted sum of: sigma_sum_norm(4.19), sigma_max_norm(1.44), novelty_min(0.29), mu_min(2.24)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.1853) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (1.4363) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (0.2917) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (2.2429) * (min(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
