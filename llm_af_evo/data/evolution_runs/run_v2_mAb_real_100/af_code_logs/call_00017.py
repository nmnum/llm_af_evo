def score_pool(context):
    """Exploitation with dynamic uncertainty and novelty boost during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign = context["campaign"]
    progress = campaign["progress"]
    stagnant_batches = campaign["stagnant_batches"]
    
    # Dynamic uncertainty weight that decreases as progress increases
    uncertainty_weight = 3.0 * (1.0 - progress)
    
    # Novelty boost only when stagnation is detected
    novelty_boost = 1.5 if stagnant_batches >= 3 else 0.0
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Combine exploitation, uncertainty, and novelty
        score = mu_sum + uncertainty_weight * sigma_norm + novelty_boost
        scores.append(score)
    
    return scores