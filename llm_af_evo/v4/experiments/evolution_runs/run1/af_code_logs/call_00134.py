def score_pool(context):
    """weighted sum of: novelty_stagnation(0.71), novelty_min(4.55), sigma_max_norm(4.49), novelty_mean(2.08)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (0.7146) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (4.5545) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (4.4873) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (2.0766) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
