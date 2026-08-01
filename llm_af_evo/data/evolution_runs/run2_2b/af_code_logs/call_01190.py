def score_pool(context):
    front = context["pareto_front"]
    ref = context["ref_point_by_name"]
    ref_arr = np.array([ref["Tm"], ref["kD"], ref["viscosity"]])
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        y = np.array([gp["Tm"]["mean"], gp["kD"]["mean"], gp["viscosity"]["mean"]])
        dominated = bool(np.any(np.all(front >= y, axis=1) & np.any(front > y, axis=1)))
        vol = float(np.prod(np.maximum(y - ref_arr, 0.0)))
        score = vol if not dominated else 0.1 * vol
        # Add exploitation term to favor candidates with high predicted objectives
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        scores.append(score + mu_sum)
    return scores