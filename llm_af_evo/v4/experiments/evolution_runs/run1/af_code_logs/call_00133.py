def score_pool(context):
    """weighted sum of: novelty_stagnation(4.36), novelty_mean(2.09), mu_min(2.40), mu_sum(1.74)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.3590) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (2.0874) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        s = s + (2.4013) * (min(gp[name]['mean'] for name in names))
        s = s + (1.7381) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
