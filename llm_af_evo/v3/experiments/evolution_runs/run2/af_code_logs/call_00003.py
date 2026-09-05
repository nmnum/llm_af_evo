def score_pool(context):
    """Weight uncertainty heavily early, decay as budget spends, boost novelty when stagnant."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign = context["campaign"]
    
    # Phase-aware weight for exploration vs exploitation
    progress = campaign["progress"]  # [0,1]
    w_exploit = min(2.0 * progress, 1.0)  # linearly increase exploit from 0 to 1
    w_uncertainty = max(1.0 - w_exploit, 0.0)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names) 
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)

        # Apply phase-aware weight to uncertainty
        score = w_exploit * mu_sum + w_uncertainty * sigma_norm
        
        scores.append(score)
    return scores