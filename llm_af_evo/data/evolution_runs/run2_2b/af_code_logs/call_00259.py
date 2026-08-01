def score_pool(context):
    """
    Approximate hypervolume-improvement: NOT exact EHVI (that needs Monte
    Carlo integration over the posterior, which the sandbox's numpy-only
    import whitelist doesn't support) — a cheap proxy using only the
    posterior mean. Candidates dominated by the current Pareto front get
    a heavily discounted score (they can still improve HV by filling in
    volume between front points, just less reliably than a non-dominated
    point does); non-dominated candidates score by their dominated volume
    relative to the reference point. This gives evolution a starting point
    that actually uses context["pareto_front"]/context["ref_point"],
    unlike the other seeds here.
    """
    front = context["pareto_front"]
    ref = context["ref_point_by_name"]
    ref_arr = np.array([ref["Tm"], ref["kD"], ref["viscosity"]])
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        y = np.array([gp["Tm"]["mean"], gp["kD"]["mean"], gp["viscosity"]["mean"]])
        dominated = bool(np.any(np.all(front >= y, axis=1) & np.any(front > y, axis=1)))
        vol = float(np.prod(np.maximum(y - ref_arr, 0.0)))
        scores.append(vol if not dominated else 0.1 * vol)
    return scores