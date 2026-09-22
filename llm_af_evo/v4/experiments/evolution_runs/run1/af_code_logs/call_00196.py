def score_pool(context):
    """weighted sum of: mu_min(1.02), mu_sum(2.45), novelty_mean(3.18)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (1.0248) * (min(gp[name]['mean'] for name in names))
        s = s + (2.4488) * (sum(gp[name]['mean'] for name in names))
        s = s + (3.1755) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
