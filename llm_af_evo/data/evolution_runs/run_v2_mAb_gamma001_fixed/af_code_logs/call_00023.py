def score_pool(context):
    """Score by predicted objective sum adjusted for uncertainty and novelty, with dynamic weighting based on progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign_progress = context["campaign"]["progress"]
    
    # Dynamic weight: explore early, exploit later
    w_exploit = 0.3 + 0.7 * (1 - np.exp(-5 * campaign_progress))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)

        # Combine exploitation and uncertainty with dynamic weight
        score_exploit_uncertainty = w_exploit * mu_sum + (1 - w_exploit) * (-sigma_norm)
        
        scores.append(score_exploit_uncertainty)
    
    return scores