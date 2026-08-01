def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean']
        ucb = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        novelty = np.linalg.norm(X_obs - cand['x'], axis=1).min()
        exploit_weight = 1.0 - progress
        explore_weight = progress * min(stagnant, 3)
        s = mu_sum + explore_weight * ucb + 0.5 * novelty * exploit_weight
        scores.append(s)
    return scores