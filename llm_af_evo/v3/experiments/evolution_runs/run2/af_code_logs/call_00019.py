def score_pool(context):
    """Exploitation-weighted uncertainty: blend of predicted mean sum and normalized std, with progress-adaptive weights."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    X_obs = context["X_obs"] 
    progress = context["campaign"]["progress"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted mean sum (exploitation)
        mu_sum = sum(gp[name]["mean"] for name in names)

        # Normalized uncertainty (exploration) 
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        # Progress-adaptive weight: early=more exploration, late=more exploitation
        w_exploit = max(0.1, 1.0 - progress * 0.8)
        
        scores.append(w_exploit * mu_sum + (1.0 - w_exploit) * sigma_norm)
    
    return scores