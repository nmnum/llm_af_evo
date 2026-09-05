def score_pool(context):
    """weighted sum of: novelty_stagnation(3.80), sigma_sum_norm(3.65), mu_min(2.35)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.8022) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (3.6473) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.3483) * (min(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
