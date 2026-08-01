def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"]/front_range["Tm"] + gp["kD"]["std"]/front_range["kD"] + gp["viscosity"]["std"]/front_range["viscosity"])
        # Blend exploitation and uncertainty with progress-dependent weight
        w = 0.5 + 0.5 * (1 - progress)  # Less exploration early, more later
        scores.append(w * mu_sum + (1 - w) * sigma_norm)
    return scores