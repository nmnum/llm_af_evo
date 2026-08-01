def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean']
        ucb = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = float(dists.min())
        w = 0.3 + 0.7 * progress
        scores.append(w * ucb + (1 - w) * mu_sum - 0.5 * novelty)
    return scores