def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        std_norm = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        max_std = max(gp['Tm']['std']/front_range['Tm'], gp['kD']['std']/front_range['kD'], gp['viscosity']['std']/front_range['viscosity'])
        s = 0.0
        s += 4.0 * std_norm
        s += 2.0 * max_std
        s += (1.0 - progress) * 1.0 * std_norm
        scores.append(s)
    return scores