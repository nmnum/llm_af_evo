def score_pool(context):
    """weighted sum of: mu_sum(2.90), novelty_stagnation(0.41), sigma_sum_norm(1.27), novelty_min(0.96)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.8977) * (sum(gp[name]['mean'] for name in names))
        s = s + (0.4061) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (1.2674) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (0.9554) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        scores.append(s)
    return scores
