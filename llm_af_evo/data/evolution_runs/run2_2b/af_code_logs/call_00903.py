def score_pool(context):
    front = context["pareto_front"]
    ref = context["ref_point_by_name"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        y = np.array([gp["Tm"]["mean"], gp["kD"]["mean"], gp["viscosity"]["mean"]])
        dominated = bool(np.any(np.all(front >= y, axis=1) & np.any(front > y, axis=1)))
        vol = float(np.prod(np.maximum(y - np.array([ref["Tm"], ref["kD"], ref["viscosity"]]), 0.0)))
        hv_score = vol if not dominated else 0.1 * vol
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = float(dists.min())
        scores.append(hv_score + 0.1 * novelty)
    return scores