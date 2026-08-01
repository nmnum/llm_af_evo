def score_pool(context):
    front = context["pareto_front"]
    ref = context["ref_point_by_name"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        y = np.array([gp["Tm"]["mean"], gp["kD"]["mean"], gp["viscosity"]["mean"]])
        dominated = bool(np.any(np.all(front >= y, axis=1) & np.any(front > y, axis=1)))
        vol = float(np.prod(np.maximum(y - np.array([ref["Tm"], ref["kD"], ref["viscosity"]]), 0.0)))
        ucb = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        score = vol if not dominated else 0.1 * vol
        score += 2.0 * ucb * max(0.0, 1.0 - progress)
        scores.append(score)
    return scores