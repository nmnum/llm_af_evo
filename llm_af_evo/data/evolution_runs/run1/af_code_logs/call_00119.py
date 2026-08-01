def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"]/front_range["Tm"] + gp["kD"]["std"]/front_range["kD"] + gp["viscosity"]["std"]/front_range["viscosity"])
        exploitation_weight = max(0.0, 1.0 - progress)
        exploration_weight = progress
        scores.append(mu_sum * exploitation_weight + 2.7532 * sigma_norm * exploration_weight)
    return scores