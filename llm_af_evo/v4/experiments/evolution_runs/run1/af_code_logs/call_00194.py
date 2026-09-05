def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(4.24), novelty_min(4.10), novelty_mean(0.13), mu_sum(3.20)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.2416) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (4.1019) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (0.1344) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        s = s + (3.2024) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
