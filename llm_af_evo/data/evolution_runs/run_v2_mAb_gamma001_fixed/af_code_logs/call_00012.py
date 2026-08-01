def score_pool(context):
    """Blend exploitation and uncertainty with progress-aware weighting, favouring exploration early and exploitation later."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    p = context["campaign"]["progress"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Early progress: more exploration (uncertainty bonus), later: more exploitation
        w_exploit = 0.3 + 0.7 * p  
        scores.append(w_exploit * mu_sum + (1 - w_exploit) * sigma_norm)
    return scores