def score_pool(context):
    """weighted sum of: novelty_stagnation(2.54), sigma_max_norm(0.90), mu_min(4.46), mu_sum(3.21), novelty_mean(1.87)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.5361) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (0.8982) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.4568) * (min(gp[name]['mean'] for name in names))
        s = s + (3.2125) * (sum(gp[name]['mean'] for name in names))
        s = s + (1.8674) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
