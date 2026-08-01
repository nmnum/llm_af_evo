def score_pool(context):
    progress = context["campaign"]["progress"]
    beta = 2.0 * (1.0 - progress)
    X_obs = context["X_obs"]
    rng = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"] / rng["Tm"] + gp["kD"]["std"] / rng["kD"]
                      + gp["viscosity"]["std"] / rng["viscosity"])
        novelty = float(np.linalg.norm(X_obs - cand["x"], axis=1).min())
        scores.append(mu_sum + beta * sigma_norm + 0.5 * novelty)
    return scores