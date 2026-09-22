def score_pool(context):
    """Balances exploitation and uncertainty using normalized GP means and standard deviations, with progress-aware weighting to shift from exploration to exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign_progress = context["campaign"]["progress"]
    
    # Adaptive weight: start with more UCB-like (uncertainty) focus early, decrease as we approach the end
    w_uncert = 0.5 + 0.3 * np.sin(campaign_progress * np.pi / 2)
    w_exploit = 1 - w_uncert
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize means and stds
        mu_norm = sum(gp[name]["mean"] / front_range[name] for name in names) 
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        score = w_exploit * mu_norm + w_uncert * sigma_norm
        
        scores.append(score)

    return scores