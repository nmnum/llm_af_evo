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
        # Exploitation term: hypervolume contribution
        exploitation = vol
        # Uncertainty term: higher uncertainty (std) gives higher score
        uncertainty = (gp["Tm"]["std"] + gp["kD"]["std"] + gp["viscosity"]["std"])
        # Combine exploitation and uncertainty with adaptive weights
        if not dominated:
            # For non-dominated candidates, weight more towards exploitation
            scores.append(0.7 * exploitation + 0.3 * uncertainty)
        else:
            # For dominated candidates, still consider uncertainty but heavily penalize
            scores.append(0.3 * exploitation + 0.7 * uncertainty)
    return scores