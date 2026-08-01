def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        ucb = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        novelty = np.linalg.norm(X_obs - cand['x'], axis=1).min()
        exploit_weight = 1.0 - progress
        explore_weight = max(0.0, 1.0 - progress) * min(stagnant, 3)
        s = 2.5 * ucb * exploit_weight + 2.0 * novelty * explore_weight
        scores.append(s)
    return scores