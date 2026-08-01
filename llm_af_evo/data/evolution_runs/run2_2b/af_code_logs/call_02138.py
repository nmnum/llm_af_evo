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
        if not dominated:
            scores.append(vol)
        else:
            # For dominated candidates, use exploitation score with penalty
            exploit_score = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
            scores.append(0.5 * exploit_score)
    return scores