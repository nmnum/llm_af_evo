def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        # Exploitation term
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        # Uncertainty term with progress-dependent weight
        sigma_norm = (gp["Tm"]["std"]/front_range["Tm"] + 
                      gp["kD"]["std"]/front_range["kD"] + 
                      gp["viscosity"]["std"]/front_range["viscosity"])
        w = 0.5 + 0.5 * progress  # weight uncertainty more early on
        ucb_score = mu_sum + w * sigma_norm
        # Novelty term
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = 1.0 / (1.0 + dists.min())
        scores.append(ucb_score + 0.5 * novelty)
    return scores