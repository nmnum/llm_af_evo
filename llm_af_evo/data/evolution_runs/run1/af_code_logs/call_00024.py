def score_pool(context):
    front = context["pareto_front"]
    ref = context["ref_point_by_name"]
    X_obs = context["X_obs"]
    progress = context["campaign"]["progress"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        y = np.array([gp["Tm"]["mean"], gp["kD"]["mean"], gp["viscosity"]["mean"]])
        
        # Hypervolume proxy score
        dominated = np.any(np.all(front >= y, axis=1) & np.any(front > y, axis=1))
        ref_arr = np.array([ref["Tm"], ref["kD"], ref["viscosity"]])
        vol = np.prod(np.maximum(y - ref_arr, 0.0))
        hv_score = vol if not dominated else 0.1 * vol
        
        # Novelty score
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = dists.min()
        
        # Blend with progress-dependent weight
        w = 0.5 + 0.5 * progress  # Start with more exploitation, shift to exploration
        scores.append(w * hv_score + (1 - w) * novelty)
    
    return scores