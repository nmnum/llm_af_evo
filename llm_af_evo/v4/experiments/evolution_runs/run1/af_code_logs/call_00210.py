def score_pool(context):
    """weighted sum of: novelty_stagnation(2.51), sigma_sum_norm(3.42), sigma_max_norm(4.30), mu_sum(0.64)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.5087) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (3.4200) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.2959) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (0.6426) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
