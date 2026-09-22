def score_pool(context):
    """weighted sum of: sigma_sum_norm(1.41), mu_min(0.42), novelty_min(0.65)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (1.4054) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (0.4244) * (min(gp[name]['mean'] for name in names))
        s = s + (0.6522) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        scores.append(s)
    return scores
