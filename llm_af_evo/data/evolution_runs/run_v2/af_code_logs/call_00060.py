def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(0.18), sigma_sum_norm(4.09), sigma_max_norm(4.57)"""
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (0.1810) * ((gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity']) * max(0.0, 1.0 - progress))
        s = s + (4.0850) * (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        s = s + (4.5725) * (max(gp['Tm']['std']/front_range['Tm'], gp['kD']['std']/front_range['kD'], gp['viscosity']['std']/front_range['viscosity']))
        scores.append(s)
    return scores
