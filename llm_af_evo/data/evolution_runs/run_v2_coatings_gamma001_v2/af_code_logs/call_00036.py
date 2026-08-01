def score_pool(context):
    """Dynamic exploitation with uncertainty and novelty, adjusted for campaign progress and stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Dynamic uncertainty weight that decreases with progress
    uncertainty_weight = max(0.3, 1.0 - progress * 0.8)
    
    # Increase exploration during stagnation
    novelty_boost = 1.0 + min(stagnant_batches, 5) * 0.1
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        score = mu_sum + uncertainty_weight * sigma_norm * novelty_boost
        scores.append(score)
    return scores