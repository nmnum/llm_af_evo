def score_pool(context):
    """weighted sum of: novelty_stagnation(3.16), sigma_sum_norm(1.04), mu_sum(1.00), novelty_mean(2.19)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.1557) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (1.0410) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (0.9959) * (sum(gp[name]['mean'] for name in names))
        s = s + (2.1921) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
