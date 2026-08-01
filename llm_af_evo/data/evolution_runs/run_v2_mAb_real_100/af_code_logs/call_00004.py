def score_pool(context):
    """Weight uncertainty heavily early, decay as budget spends, with novelty boost during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign = context["campaign"]
    progress = campaign["progress"]
    stagnant_batches = campaign["stagnant_batches"]
    
    # Phase-aware decay of uncertainty weight
    uncertainty_weight = 2.0 * (1.0 - progress)
    
    # Novelty boost if stagnant
    novelty_boost = 0.0 if stagnant_batches < 3 else 1.0
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Combine exploitation and uncertainty with phase-aware weighting
        score = mu_sum + uncertainty_weight * sigma_norm + novelty_boost
        scores.append(score)
    
    return scores