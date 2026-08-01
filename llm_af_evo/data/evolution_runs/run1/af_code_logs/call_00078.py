def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"] / front_range["Tm"]
                      + gp["kD"]["std"] / front_range["kD"]
                      + gp["viscosity"]["std"] / front_range["viscosity"])
        # Blend exploitation and uncertainty with progress-aware weighting
        exploit_weight = max(0.5, 1.0 - progress * 0.5)
        scores.append(exploit_weight * mu_sum + (1 - exploit_weight) * sigma_norm)
    return scores