def score_pool(context):
    """weighted sum of: sigma_sum_norm_early(0.11), mu_sum(1.68)"""
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        s = 0.0
        s = s + (0.1134) * ((gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity']) * max(0.0, 1.0 - progress))
        s = s + (1.6845) * (gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean'])
        scores.append(s)
    return scores
