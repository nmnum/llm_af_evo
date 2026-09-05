def score_pool(context):
    """weighted sum of: novelty_stagnation(3.40), mu_sum(2.03)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (3.3951) * (float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * min(stagnant, 5))
        s = s + (2.0252) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
