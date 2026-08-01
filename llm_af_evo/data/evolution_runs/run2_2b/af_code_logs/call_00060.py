def score_pool(context):
    """Fixed-weight UCB-style: predicted objective sum plus a beta*sigma bonus."""
    beta = 2.0
    rng = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"] / rng["Tm"] + gp["kD"]["std"] / rng["kD"]
                      + gp["viscosity"]["std"] / rng["viscosity"])
        scores.append(mu_sum + beta * sigma_norm)
    return scores