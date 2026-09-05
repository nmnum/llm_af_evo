def score_pool(context):
    """weighted sum of: novelty_stagnation(4.02), sigma_sum_norm(1.47), sigma_max_norm(4.65), mu_sum(2.78)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.0216) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (1.4659) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.6482) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.7813) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
