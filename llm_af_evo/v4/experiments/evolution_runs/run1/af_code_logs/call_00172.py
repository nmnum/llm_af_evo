def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(2.88), novelty_stagnation(3.35), sigma_sum_norm(1.26), mu_min(1.90)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.8836) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (3.3471) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (1.2642) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (1.9015) * (min(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
