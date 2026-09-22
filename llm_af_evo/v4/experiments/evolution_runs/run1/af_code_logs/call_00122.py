def score_pool(context):
    """weighted sum of: mu_sum(2.06), sigma_sum_norm(4.46), sigma_max_norm(4.86), novelty_mean(0.45)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.0636) * (sum(gp[name]['mean'] for name in names))
        s = s + (4.4563) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.8552) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (0.4499) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
