def score_pool(context):
    """
    Phase-aware: explore early (high sigma weight when little of the
    budget is spent), exploit late, with an extra novelty kick if the
    campaign has been stagnant. Demonstrates the kind of strategy a purely
    per-candidate (x, mu, sigma) -> score function cannot express.
    """
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    beta = 3.0 * (1.0 - progress)
    stagnation_boost = 1.0 + 0.3 * min(stagnant, 5)
    X_obs = context["X_obs"]
    rng = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"] / rng["Tm"] + gp["kD"]["std"] / rng["kD"]
                      + gp["viscosity"]["std"] / rng["viscosity"])
        novelty = float(np.linalg.norm(X_obs - cand["x"], axis=1).min())
        scores.append(mu_sum + beta * sigma_norm + stagnation_boost * 0.3 * novelty)
    return scores