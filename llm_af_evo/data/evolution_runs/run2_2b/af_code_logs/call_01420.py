def score_pool(context):
    progress = context["campaign"]["progress"]
    front = context["pareto_front"]
    ref = context["ref_point_by_name"]
    rng = context["pareto_front_range"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        y = np.array([gp["Tm"]["mean"], gp["kD"]["mean"], gp["viscosity"]["mean"]])
        dominated = bool(np.any(np.all(front >= y, axis=1) & np.any(front > y, axis=1)))
        vol = float(np.prod(np.maximum(y - np.array([ref["Tm"], ref["kD"], ref["viscosity"]]), 0.0)))
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"] / rng["Tm"] + gp["kD"]["std"] / rng["kD"]
                      + gp["viscosity"]["std"] / rng["viscosity"])
        novelty = float(np.linalg.norm(X_obs - cand["x"], axis=1).min())
        beta = 2.0 * (1.0 - progress)
        scores.append((vol if not dominated else 0.1 * vol) + beta * sigma_norm + 0.5 * novelty)
    return scores