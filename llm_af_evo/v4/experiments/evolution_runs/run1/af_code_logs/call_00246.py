def score_pool(context):
    """weighted sum of: novelty_min(2.69), sigma_sum_norm(3.56), mu_min(4.16)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.6857) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (3.5626) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.1569) * (min(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
