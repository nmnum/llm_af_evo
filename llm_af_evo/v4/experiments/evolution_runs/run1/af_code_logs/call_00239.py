def score_pool(context):
    """weighted sum of: novelty_stagnation(3.89), sigma_sum_norm_early(0.92)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.8948) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (0.9224) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        scores.append(s)
    return scores
