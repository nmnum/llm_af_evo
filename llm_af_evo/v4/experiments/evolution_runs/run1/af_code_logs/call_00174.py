def score_pool(context):
    """weighted sum of: mu_min(2.00), mu_sum(2.10)"""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.0045) * (min(gp[name]['mean'] for name in names))
        s = s + (2.0978) * (sum(gp[name]['mean'] for name in names))
        scores.append(s)
    return scores
