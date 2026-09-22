def score_pool(context):
    """weighted sum of: sigma_max_norm(1.19), novelty_stagnation(1.94), mu_min(0.21), mu_sum(1.84)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (1.1917) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (1.9356) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (0.2108) * (min(gp[name]['mean'] for name in names))
        s = s + (1.8439) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
