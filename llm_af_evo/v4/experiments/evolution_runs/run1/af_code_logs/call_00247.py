def score_pool(context):
    """weighted sum of: novelty_min(2.90), sigma_sum_norm(1.55), sigma_sum_norm_early(2.16), mu_min(2.88), novelty_mean(3.87)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.9038) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (1.5504) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.1601) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (2.8764) * (min(gp[name]['mean'] for name in names))
        s = s + (3.8737) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
