def score_pool(context):
    """Weight uncertainty heavily early, decay as budget spends, exploit more later, with novelty boost during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign = context["campaign"]
    progress = campaign["progress"]
    stagnant_batches = campaign["stagnant_batches"]
    
    # Decay factor for uncertainty term: higher early, lower later
    uncertainty_weight = 1.0 - progress
    
    # Novelty boost when stagnant
    novelty_boost = 1.0 + min(stagnant_batches, 5) * 0.2
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Combine exploitation and uncertainty with decay and novelty
        score = mu_sum + uncertainty_weight * sigma_norm * novelty_boost
        scores.append(score)
    
    return scores