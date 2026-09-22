def score_pool(context):
    """weighted sum of: sigma_sum_norm(0.98), novelty_min(4.50), sigma_max_norm(3.87)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (0.9793) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.4960) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (3.8721) * (max(gp[name]['std']/front_range[name] for name in names))
        scores.append(s)
    return scores
