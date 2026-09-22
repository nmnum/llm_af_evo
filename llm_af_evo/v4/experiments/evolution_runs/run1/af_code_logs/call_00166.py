def score_pool(context):
    """weighted sum of: novelty_stagnation(1.80), novelty_min(2.86), mu_min(3.94)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (1.8014) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (2.8594) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (3.9399) * (min(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
