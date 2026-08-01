def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        ucb = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        novelty = np.linalg.norm(X_obs - cand['x'], axis=1).min()
        exploit_weight = 1.0 - min(0.8, progress * 2.0)
        explore_weight = 0.5 + 0.5 * min(1.0, progress * 2.0)
        s = exploit_weight * mu_sum + explore_weight * ucb + 0.1 * novelty * min(stagnant, 3) * (1.0 - progress)
        scores.append(s)
    return scores