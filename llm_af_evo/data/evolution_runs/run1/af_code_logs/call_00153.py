def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"] / front_range["Tm"] +
                      gp["kD"]["std"] / front_range["kD"] +
                      gp["viscosity"]["std"] / front_range["viscosity"])
        # Blend exploitation and uncertainty with progress-dependent weighting
        w_exploit = max(0.3, 1.0 - progress * 0.7)
        score = w_exploit * mu_sum + (1 - w_exploit) * sigma_norm * 2.0
        scores.append(score)
    return scores