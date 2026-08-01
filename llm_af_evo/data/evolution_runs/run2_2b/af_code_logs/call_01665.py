def score_pool(context):
    front = context["pareto_front"]
    ref = context["ref_point_by_name"]
    ref_arr = np.array([ref["Tm"], ref["kD"], ref["viscosity"]])
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        y = np.array([gp["Tm"]["mean"], gp["kD"]["mean"], gp["viscosity"]["mean"]])
        dominated = np.any(np.all(front >= y, axis=1) & np.any(front > y, axis=1))
        vol = np.prod(np.maximum(y - ref_arr, 0.0))
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        scores.append(mu_sum + (0.1 if dominated else 1.0) * vol)
    return scores