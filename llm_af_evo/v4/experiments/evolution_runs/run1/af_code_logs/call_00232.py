def score_pool(context):
    """weighted sum of: novelty_stagnation(2.30), sigma_sum_norm(1.36), mu_min(0.47), sigma_max_norm(0.13)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.2995) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (1.3575) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (0.4658) * (min(gp[name]['mean'] for name in names))
        s = s + (0.1259) * (max(gp[name]['std']/front_range[name] for name in names))
        scores.append(s)
    return scores
