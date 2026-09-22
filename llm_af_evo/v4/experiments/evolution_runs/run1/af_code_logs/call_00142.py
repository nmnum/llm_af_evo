def score_pool(context):
    """weighted sum of: mu_sum(4.22), novelty_min(3.08), novelty_mean(4.04)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.2174) * (sum(gp[name]['mean'] for name in names))
        s = s + (3.0792) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (4.0396) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
