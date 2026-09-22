def score_pool(context):
    """Blend UCB-style exploitation with uncertainty and novelty-aware selection."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign = context["campaign"]
    
    # Phase-aware weight for exploration vs exploitation  
    progress = campaign["progress"]  # [0,1]
    w_exploit = min(2.0 * progress, 1.0) 
    w_uncertainty = max(1.0 - w_exploit, 0.0)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names) 
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)

        # Apply phase-aware weight to uncertainty
        score = w_exploit * mu_sum + w_uncertainty * sigma_norm
        
        # Add novelty bonus when stagnant (distance from nearest observed point)
        if campaign["stagnant_batches"] > 0:
            x_observed = context["X_obs"]
            dist_to_nearest = np.min(np.linalg.norm(x_observed - cand["x"], axis=1))
            score += max(2.76 * (1.0 / (dist_to_nearest + 1e-8) - 1), 0)
        
        scores.append(score)
    return scores