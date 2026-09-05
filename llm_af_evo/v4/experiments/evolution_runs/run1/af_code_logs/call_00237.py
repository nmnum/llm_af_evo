def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(4.90), novelty_min(1.59), sigma_max_norm(2.00), mu_sum(3.57)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.9029) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (1.5916) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (1.9952) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (3.5703) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
