def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        ucb_term = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        s = 0.0
        s = s + 0.181 * ucb_term * max(0.0, 1.0 - progress)
        s = s + 4.085 * ucb_term
        s = s + 4.573 * max(gp['Tm']['std']/front_range['Tm'], gp['kD']['std']/front_range['kD'], gp['viscosity']['std']/front_range['viscosity'])
        scores.append(s)
    return scores