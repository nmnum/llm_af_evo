def score_pool(context):
    """Exploitation with uncertainty and novelty."""
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"] / context["pareto_front_range"]["Tm"]
                      + gp["kD"]["std"] / context["pareto_front_range"]["kD"]
                      + gp["viscosity"]["std"] / context["pareto_front_range"]["viscosity"])
        # Distance to nearest observed point
        dist = np.min(np.sum((context["X_obs"] - cand["x"])**2, axis=1))
        novelty = 1.0 / (1.0 + dist)
        scores.append(mu_sum + 0.5 * sigma_norm + 0.5 * novelty)
    return scores