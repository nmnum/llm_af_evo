def score_pool(context):
    """Exploitation-weighted uncertainty with hypervolume-aware normalization and progress-adaptive blending."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized uncertainty
        sigma_norm = sum(gp[name]['std'] / front_range[name] for name in names)
        
        # Progress-adaptive exploitation vs exploration trade-off
        exploit_weight = max(0.0, 1.0 - progress * 2)  # Decrease exploitation as we go further
        
        # Combined score: blend of uncertainty and novelty (with adaptive weights)
        s = sigma_norm * exploit_weight + \
            float(np.linalg.norm(X_obs - cand['x'], axis=1).min()) * (1 - exploit_weight)

        scores.append(s)
    return scores