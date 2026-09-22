def score_pool(context):
    """weighted sum of: sigma_max_norm(2.60), novelty_mean(1.29), mu_min(1.55)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.6037) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (1.2939) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        s = s + (1.5517) * (min(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
