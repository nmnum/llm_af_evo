def score_pool(context):
    """weighted sum of: mu_sum(1.00)"""
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (1.0000) * (gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean'])
        scores.append(s)
    return scores
