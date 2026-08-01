def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        # Exploitation + uncertainty tradeoff
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"]/front_range["Tm"] + 
                      gp["kD"]["std"]/front_range["kD"] + 
                      gp["viscosity"]["std"]/front_range["viscosity"])
        # Adaptive weighting: more exploitation early, more exploration later
        w_exploit = 1.0 - min(1.0, progress * 2.0)
        score = w_exploit * mu_sum + (1.0 - w_exploit) * sigma_norm * 2.0
        # Add novelty bonus
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = 1.0 / (1.0 + dists.min())
        score += novelty * 0.5
        scores.append(score)
    return scores