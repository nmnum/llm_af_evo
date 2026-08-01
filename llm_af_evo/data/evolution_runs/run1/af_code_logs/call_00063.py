def score_pool(context):
    """Exploitation with uncertainty and novelty."""
    front_range = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"] / front_range["Tm"]
                      + gp["kD"]["std"] / front_range["kD"]
                      + gp["viscosity"]["std"] / front_range["viscosity"])
        # Novelty term: inverse distance to nearest observed point
        x = cand["x"]
        if len(context["X_obs"]) == 0:
            novelty = 1.0
        else:
            distances = np.linalg.norm(context["X_obs"] - x, axis=1)
            novelty = 1.0 / (np.min(distances) + 1e-8)
        scores.append(mu_sum + 1.0 * sigma_norm + 0.1 * novelty)
    return scores