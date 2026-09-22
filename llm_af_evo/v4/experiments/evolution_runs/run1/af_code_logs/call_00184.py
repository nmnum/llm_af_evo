def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(1.02), novelty_stagnation(4.04), sigma_max_norm(3.22), mu_min(1.50)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (1.0188) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (4.0415) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (3.2183) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (1.5034) * (min(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
