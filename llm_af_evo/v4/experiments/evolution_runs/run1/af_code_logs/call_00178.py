def score_pool(context):
    """weighted sum of: novelty_min(3.49), sigma_max_norm(3.93), mu_sum(4.52), novelty_mean(4.25)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.4905) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (3.9282) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.5241) * (sum(gp[name]['mean'] for name in names))
        s = s + (4.2516) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
