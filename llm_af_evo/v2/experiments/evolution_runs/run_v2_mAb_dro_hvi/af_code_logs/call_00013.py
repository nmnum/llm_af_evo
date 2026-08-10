def score_pool(context):
    """Exploitation with dynamic uncertainty weighting and novelty boost during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Dynamic weight for uncertainty: higher early, lower later
    w_uncertainty = max(0.5 * (1 - progress), 0.1)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Combine exploitation and uncertainty
        score = w_uncertainty * sigma_norm + (1 - w_uncertainty) * mu_sum
        
        # Add novelty bonus if stagnant to encourage exploration
        if stagnant_batches >= 3:
            dist_to_observed = np.min(np.linalg.norm(cand["x"] - context["X_obs"], axis=1))
            novel_bonus = max(0, 2.0 / (dist_to_observed + 1e-6)) 
            score += novel_bonus
            
        scores.append(score)
    
    return scores