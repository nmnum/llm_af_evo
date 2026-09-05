def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(1.06), novelty_stagnation(1.52), sigma_sum_norm(4.60), novelty_min(3.02), mu_sum(1.55), novelty_mean(4.09)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (1.0646) * (sum(gp[name]['std']/front_range[name] for name in names) * max(0.0, 1.0 - progress))
        s = s + (1.5202) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (4.6040) * (sum(gp[name]['std']/front_range[name] for name in names))
        s = s + (3.0182) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (1.5489) * (sum(gp[name]['mean'] for name in names))
        s = s + (4.0906) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
