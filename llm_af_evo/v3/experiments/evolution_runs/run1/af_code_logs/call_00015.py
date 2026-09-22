def score_pool(context):
    """Weighted combination of uncertainty-based exploitation and novelty reward, with progressive exploration emphasis."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Uncertainty component: normalized standard deviation
        sigma_norm = sum(gp[name]['std'] / front_range[name] for name in names)
        
        # Exploitation component: predicted mean (already flipped to be higher-is-better)
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Progressive weighting between exploration and exploitation
        w_exploit = 1.0 - max(0.0, progress * 2.0 - 1.0)  # Decrease exploit weight as we go
        
        score = (w_exploit * mu_sum + 
                (1.0 - w_exploit) * sigma_norm)
        
        scores.append(score)

    return scores