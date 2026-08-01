def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        uncertainty = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        novelty = float(np.linalg.norm(X_obs - cand['x'], axis=1).min())
        s = 2.0 * uncertainty + 3.0 * novelty * min(stagnant, 5) * max(0.0, 1.0 - progress)
        scores.append(s)
    return scores