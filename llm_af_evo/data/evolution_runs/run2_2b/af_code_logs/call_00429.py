def score_pool(context):
    """Combine hypervolume improvement estimate with novelty."""
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    X_obs = context["X_obs"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Estimate HV improvement using GP predictions
        mu = np.array([gp["Tm"]["mean"], gp["kD"]["mean"], gp["viscosity"]["mean"]])
        sigma = np.array([gp["Tm"]["std"], gp["kD"]["std"], gp["viscosity"]["std"]])
        
        # Normalize by front range for relative scale
        mu_norm = (mu - ref_point) / np.array(list(front_range.values()))
        sigma_norm = sigma / np.array(list(front_range.values()))
        
        # Simple hypervolume estimate: product of distances to reference point
        hv_improvement = np.prod(np.maximum(mu_norm, 0))
        
        # Novelty term: distance to nearest observed point
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = 1.0 / (1e-8 + dists.min())
        
        # Balance exploitation (HV) and exploration (novelty)
        scores.append(hv_improvement + 0.5 * novelty)
    
    return scores