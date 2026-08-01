def score_pool(context):
    """weighted sum of: novelty_min(2.76), mu_sum(1.00), novelty_stagnation(5.66)"""
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.7638) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()))
        s = s + (1.0000) * (gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean'])
        s = s + (5.6578) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        scores.append(s)
    return scores
