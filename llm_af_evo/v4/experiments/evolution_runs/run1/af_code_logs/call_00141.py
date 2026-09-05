def score_pool(context):
    """weighted sum of: novelty_stagnation(2.88), sigma_sum_norm(1.99), sigma_max_norm(1.53), mu_min(2.20), novelty_mean(1.55)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.8801) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (1.9924) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (1.5252) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.2018) * (min(gp[name]['mean'] for name in names))
        s = s + (1.5512) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
