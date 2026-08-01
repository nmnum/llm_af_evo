def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_norm = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        exploit_weight = max(0.0, 1.0 - progress)
        explore_weight = progress
        s = explore_weight * sigma_norm * (0.1134 + 2.0578 * exploit_weight)
        scores.append(s)
    return scores