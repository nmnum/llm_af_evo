def score_pool(context):
    """weighted sum of: novelty_min(1.54), sigma_max_norm(2.45), mu_sum(3.86)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (1.5377) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (2.4489) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (3.8566) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
