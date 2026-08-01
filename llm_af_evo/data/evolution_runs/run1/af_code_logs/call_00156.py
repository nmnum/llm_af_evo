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
        exploit_weight = max(0.0, 1.0 - progress**2)
        explore_weight = progress * (1.0 - exploit_weight)
        s = exploit_weight * mu_sum + explore_weight * ucb + 0.5 * novelty
        scores.append(s)
    return scores