def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        sigma_norm = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        exploitation = (gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean']) / 3.0
        scores.append(0.5 * sigma_norm * max(0.0, 1.0 - progress) + 0.5 * exploitation)
    return scores