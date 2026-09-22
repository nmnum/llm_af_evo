def score_pool(context):
    """Exploitation-weighted uncertainty: blend of predicted mean sum and normalized std, with progress-adaptive weights."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    # Adaptive weight for exploitation vs exploration
    w_exploit = 1.0 - max(0.0, min(1.0, (2 * progress) ** 3))
    w_uncertain = 1.0 - w_exploit
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)
        
        score = (w_exploit * mu_sum) + (w_uncertain * sigma_norm)
        scores.append(score)
    
    return scores