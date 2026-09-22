def score_pool(context):
    """weighted sum of: sigma_max_norm(4.88), mu_sum(1.43), novelty_mean(2.05)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (4.8760) * (max(gp[name]['std']/front_range[name] for name in names))
        s = s + (1.4299) * (sum(gp[name]['mean'] for name in names))
        s = s + (2.0498) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).mean()))
        scores.append(s)
    return scores
