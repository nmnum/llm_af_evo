def score_pool(context):
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    front = context["pareto_front"]
    ref = context["ref_point_by_name"]
    ref_arr = np.array([ref["Tm"], ref["kD"], ref["viscosity"]])
    X_obs = context["X_obs"]
    rng = context["pareto_front_range"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        y = np.array([gp["Tm"]["mean"], gp["kD"]["mean"], gp["viscosity"]["mean"]])
        dominated = bool(np.any(np.all(front >= y, axis=1) & np.any(front > y, axis=1)))
        vol = float(np.prod(np.maximum(y - ref_arr, 0.0)))
        hv_score = vol if not dominated else 0.1 * vol
        
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_norm = (gp["Tm"]["std"] / rng["Tm"] + gp["kD"]["std"] / rng["kkD"]
                      + gp["viscosity"]["std"] / rng["viscosity"])
        beta = 2.0 * (1.0 - progress)
        stagnation_boost = 1.0 + 0.5 * min(stagnant, 5)
        
        novelty = float(np.linalg.norm(X_obs - cand["x"], axis=1).min())
        
        scores.append(hv_score + beta * sigma_norm + stagnation_boost * 0.2 * novelty)
    
    return scores