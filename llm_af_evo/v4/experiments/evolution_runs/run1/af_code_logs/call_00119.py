def score_pool(context):
    """weighted sum of: novelty_stagnation(3.35), sigma_max_norm(3.77), mu_min(4.58), mu_sum(1.44), novelty_mean(3.78)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.3521) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (3.7748) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.5841) * (min(gp[name]['mean'] for name in names))
        s = s + (1.4413) * (sum(gp[name]['mean'] for name in names))
        s = s + (3.7825) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
