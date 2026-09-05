def score_pool(context):
    """weighted sum of: novelty_stagnation(1.78), novelty_min(0.18), sigma_max_norm(1.52), mu_sum(4.46), novelty_mean(4.83)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (1.7810) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (0.1802) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (1.5175) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (4.4610) * (sum(gp[name]['mean'] for name in names))
        s = s + (4.8339) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
