def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_norm = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        exploit_weight = max(0.0, 1.0 - progress)
        explore_weight = 1.0
        s = (2.1712 * sigma_norm * explore_weight * exploit_weight)
        scores.append(s)
    return scores