def score_pool(context):
    """Combine hypervolume proxy with novelty and uncertainty exploration."""
    front = context["pareto_front"]
    ref = context["ref_point_by_name"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        y = np.array([gp["Tm"]["mean"], gp["kD"]["mean"], gp["viscosity"]["mean"]])
        # Hypervolume score
        vol = float(np.prod(np.maximum(y - np.array([ref["Tm"], ref["kD"], ref["viscosity"]]), 0.0)))
        # Dominance check
        dominated = bool(np.any(np.all(front >= y, axis=1) & np.any(front > y, axis=1)))
        hv_score = vol if not dominated else 0.1 * vol
        # Uncertainty term (normalized by front range)
        front_range = context["pareto_front_range"]
        uncertainty = (gp["Tm"]["std"] / front_range["Tm"] + 
                       gp["kD"]["std"] / front_range["kD"] + 
                       gp["viscosity"]["std"] / front_range["viscosity"])
        # Novelty term
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = 1.0 / (1e-8 + dists.min())
        # Combine: exploitation + uncertainty + novelty
        scores.append(hv_score + 2.0 * uncertainty + 0.5 * novelty)
    return scores