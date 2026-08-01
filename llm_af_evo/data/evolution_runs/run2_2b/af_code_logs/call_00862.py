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
        score = vol if not dominated else 0.1 * vol
        # Add uncertainty bonus for non-dominated candidates
        if not dominated:
            ucb = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
            ucb += (gp["Tm"]["std"] + gp["kD"]["std"] + gp["viscosity"]["std"])
            score += 0.5 * ucb
        scores.append(score)
    return scores