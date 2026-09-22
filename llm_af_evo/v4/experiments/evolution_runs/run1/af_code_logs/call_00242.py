def score_pool(context):
    """weighted sum of: novelty_stagnation(2.83), novelty_min(0.54), mu_sum(3.83), novelty_mean(1.13)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.8324) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (0.5392) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (3.8291) * (sum(gp[name]['mean'] for name in names))
        s = s + (1.1344) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
