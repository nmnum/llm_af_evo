def score_pool(context):
    """weighted sum of: sigma_sum_norm(2.17), mu_sum(1.00), sigma_sum_norm_early(4.64)"""
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (2.1712) * (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        s = s + (1.0000) * (gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean'])
        s = s + (4.6444) * ((gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity']) * max(0.0, 1.0 - progress))
        scores.append(s)
    return scores
