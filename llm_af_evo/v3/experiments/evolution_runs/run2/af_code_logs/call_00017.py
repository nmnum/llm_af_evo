def score_pool(context):
    """Exploitation-weighted uncertainty: blend of predicted mean sum and normalized std, with progress-adaptive weights."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    X_obs = context["X_obs"]
    
    # Adaptive weight for exploitation vs exploration
    progress = context["campaign"]["progress"] 
    w_exploit = 0.5 + 0.3 * (1 - np.exp(-2*progress))   # Start low, ramp up to ~0.8
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names) 
        
        # Combine exploitation and uncertainty with adaptive weights
        score = w_exploit * mu_sum - (1-w_exploit) * sigma_norm
        
        scores.append(score)

    return scores