def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(3.25), novelty_stagnation(0.95), novelty_mean(2.70), novelty_min(4.14)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.2453) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (0.9464) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (2.7033) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        s = s + (4.1359) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        scores.append(s)
    return scores
