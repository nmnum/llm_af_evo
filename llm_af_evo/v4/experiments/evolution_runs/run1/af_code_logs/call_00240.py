def score_pool(context):
    """weighted sum of: sigma_max_norm(1.63), sigma_sum_norm(2.27), mu_sum(2.59), mu_min(0.94), novelty_mean(4.84)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (1.6254) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.2656) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.5890) * (sum(gp[name]['mean'] for name in names))
        s = s + (0.9407) * (min(gp[name]['mean'] for name in names))
        s = s + (4.8431) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
