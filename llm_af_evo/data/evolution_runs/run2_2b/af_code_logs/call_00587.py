def score_pool(context):
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    X_obs = context["X_obs"]
    rng = context["pareto_front_range"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"] / rng["Tm"] + gp["kD"]["std"] / rng["kkD"]
                      + gp["viscosity"]["std"] / rng["viscosity"])
        
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = dists.min()
        
        beta = 3.0 * (1.0 - progress)
        stagnation_boost = 1.0 + 0.3 * min(stagnant, 5)
        
        scores.append(mu_sum + beta * sigma_norm + stagnation_boost * 0.3 * novelty)
    
    return scores